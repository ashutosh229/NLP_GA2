"""One-time data preparation: run this first (on each machine, or run once and
rsync the DATA_DIR from common.py to the other desktop to skip repeating it).

Produces, under $DATA_DIR:
  vocab.pkl, embedding_matrix.npy                     (BiGRU inputs)
  classic_{train,val,test}_ids.npy / _labels.npy       (BiGRU-encoded splits)
  roberta_256/, roberta_128/                           (tokenized arrow datasets)
"""
import common  # noqa: F401  (sets HF_HOME before transformers/datasets import)
import os, re, pickle, collections, zipfile, urllib.request, time
import numpy as np
from datasets import load_dataset
from transformers import RobertaTokenizerFast
import spacy
from tqdm import tqdm

from common import (DATA_DIR, FASTTEXT_LOCAL, FASTTEXT_VEC_URL, FASTTEXT_DIR,
                     MAX_SEQ_LEN_CLASSIC, MIN_FREQ, EMBED_DIM, PAD_TOKEN, UNK_TOKEN,
                     DATASET_NAME, ROBERTA_MODEL)

HTML_RE  = re.compile(r'<[^>]+>')
URL_RE   = re.compile(r'https?://\S+|www\.\S+')
SPACE_RE = re.compile(r'\s+')


def clean_text(text: str) -> str:
    text = HTML_RE.sub(' ', text)
    text = URL_RE.sub(' ', text)
    return SPACE_RE.sub(' ', text.lower()).strip()


def get_text(sample) -> str:
    if sample.get('text'):
        return sample['text'].strip()
    return ((sample.get('review_title') or '') + ' ' + (sample.get('review_body') or '')).strip()


def main():
    t_start = time.time()
    print('Loading dataset ...')
    dataset  = load_dataset(DATASET_NAME)
    train_ds, val_ds, test_ds = dataset['train'], dataset['validation'], dataset['test']
    print(dataset)

    # ── Classic preprocessing (BiGRU) ──────────────────────────────────────
    print('\n[1/4] spaCy tokenization for classic pipeline (BiGRU) ...')
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

    def tok_batch(texts):
        return [[t.text for t in doc if not t.is_space]
                for doc in nlp.pipe((clean_text(t) for t in texts), batch_size=256, n_process=8)]

    train_texts = [get_text(s) for s in train_ds]
    val_texts   = [get_text(s) for s in val_ds]
    test_texts  = [get_text(s) for s in test_ds]

    print(f'  tokenizing {len(train_texts):,} train texts ...')
    train_toks = tok_batch(train_texts)
    print(f'  tokenizing {len(val_texts):,} val texts ...')
    val_toks   = tok_batch(val_texts)
    print(f'  tokenizing {len(test_texts):,} test texts ...')
    test_toks  = tok_batch(test_texts)

    word_freq = collections.Counter()
    for toks in train_toks:
        word_freq.update(toks)
    vocab    = [PAD_TOKEN, UNK_TOKEN] + [w for w, c in word_freq.items() if c >= MIN_FREQ]
    word2idx = {w: i for i, w in enumerate(vocab)}
    print(f'  vocabulary size: {len(vocab):,}')

    def encode_tokens(toks, max_len=MAX_SEQ_LEN_CLASSIC):
        unk = word2idx[UNK_TOKEN]
        ids = [word2idx.get(t, unk) for t in toks[:max_len]]
        if len(ids) == 0:
            ids = [unk]
        ids += [0] * (max_len - len(ids))
        return ids

    train_ids = np.array([encode_tokens(t) for t in train_toks], dtype=np.int32)
    val_ids   = np.array([encode_tokens(t) for t in val_toks],   dtype=np.int32)
    test_ids  = np.array([encode_tokens(t) for t in test_toks],  dtype=np.int32)
    train_lbl = np.array(train_ds['label'], dtype=np.int64)
    val_lbl   = np.array(val_ds['label'],   dtype=np.int64)
    test_lbl  = np.array(test_ds['label'],  dtype=np.int64)

    pickle.dump({'vocab': vocab, 'word2idx': word2idx}, open(f'{DATA_DIR}/vocab.pkl', 'wb'))
    np.save(f'{DATA_DIR}/classic_train_ids.npy', train_ids)
    np.save(f'{DATA_DIR}/classic_val_ids.npy',   val_ids)
    np.save(f'{DATA_DIR}/classic_test_ids.npy',  test_ids)
    np.save(f'{DATA_DIR}/classic_train_labels.npy', train_lbl)
    np.save(f'{DATA_DIR}/classic_val_labels.npy',   val_lbl)
    np.save(f'{DATA_DIR}/classic_test_labels.npy',  test_lbl)

    # ── fastText embedding matrix ───────────────────────────────────────────
    print('\n[2/4] Building fastText embedding matrix ...')
    if not os.path.exists(FASTTEXT_LOCAL):
        print('  downloading fastText vectors from Meta AI (~1.6GB zip) ...')
        zip_path = f'{FASTTEXT_DIR}/crawl-300d-2M.vec.zip'
        urllib.request.urlretrieve(FASTTEXT_VEC_URL, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(FASTTEXT_DIR)
        os.remove(zip_path)
    else:
        print('  using cached vectors:', FASTTEXT_LOCAL)

    embedding_matrix = np.random.normal(0, 0.1, (len(vocab), EMBED_DIM)).astype(np.float32)
    embedding_matrix[0] = 0.0
    hit = 0
    with open(FASTTEXT_LOCAL, encoding='utf-8', errors='ignore') as fh:
        for line in tqdm(fh, desc='  loading vectors'):
            parts = line.rstrip().split(' ')
            if len(parts) != EMBED_DIM + 1:
                continue
            word = parts[0]
            if word in word2idx:
                embedding_matrix[word2idx[word]] = np.array(parts[1:], dtype=np.float32)
                hit += 1
    embedding_matrix = np.nan_to_num(embedding_matrix, copy=False)
    print(f'  coverage: {hit}/{len(vocab)} = {hit/len(vocab)*100:.1f}%')
    np.save(f'{DATA_DIR}/embedding_matrix.npy', embedding_matrix)

    # ── RoBERTa tokenization (256 for main model, 128 for ablation B) ──────
    print('\n[3/4] RoBERTa tokenization (max_len=256, main model) ...')
    tokenizer = RobertaTokenizerFast.from_pretrained(ROBERTA_MODEL)

    def tokenize_split(ds, max_len, desc=''):
        drop = [c for c in ds.column_names if c != 'label']
        out = ds.map(lambda ex: tokenizer([t or '' for t in ex['text']], max_length=max_len,
                                           truncation=True, padding=False),
                     batched=True, batch_size=1000, remove_columns=drop, desc=desc)
        return out

    for max_len, tag in [(256, 'roberta_256'), (128, 'roberta_128')]:
        print(f'\n[4/4] tokenizing for max_len={max_len} ...')
        tr = tokenize_split(train_ds, max_len, f'train-{max_len}')
        va = tokenize_split(val_ds,   max_len, f'val-{max_len}')
        te = tokenize_split(test_ds,  max_len, f'test-{max_len}')
        tr.save_to_disk(f'{DATA_DIR}/{tag}/train')
        va.save_to_disk(f'{DATA_DIR}/{tag}/validation')
        te.save_to_disk(f'{DATA_DIR}/{tag}/test')

    print(f'\nDone in {time.time()-t_start:.0f}s. Cached under {DATA_DIR}')


if __name__ == '__main__':
    main()
