import os
import re
import sys
import json
import pickle
import unicodedata
from collections import Counter
from typing import List, Tuple, Dict, Optional
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

# Ensure UTF-8 output
sys.stdout.reconfigure(encoding="utf-8")

PAD_TOKEN = "<PAD>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"
UNK_TOKEN = "<UNK>"

PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3

class Vocabulary:
    def __init__(self, name: str, min_freq: int = 2):
        self.name = name
        self.min_freq = min_freq
        self.word2idx = {PAD_TOKEN: PAD_IDX, SOS_TOKEN: SOS_IDX, EOS_TOKEN: EOS_IDX, UNK_TOKEN: UNK_IDX}
        self.idx2word = {PAD_IDX: PAD_TOKEN, SOS_IDX: SOS_TOKEN, EOS_IDX: EOS_TOKEN, UNK_IDX: UNK_TOKEN}
        self.word_counts = Counter()
        self.n_words = 4

    def add_sentence(self, tokens: List[str]):
        for token in tokens:
            self.word_counts[token] += 1

    def build_vocab(self):
        for word, count in self.word_counts.most_common():
            if count >= self.min_freq:
                if word not in self.word2idx:
                    self.word2idx[word] = self.n_words
                    self.idx2word[self.n_words] = word
                    self.n_words += 1

    def encode(self, tokens: List[str], add_sos_eos: bool = True) -> List[int]:
        ids = []
        if add_sos_eos:
            ids.append(SOS_IDX)
        for token in tokens:
            ids.append(self.word2idx.get(token, UNK_IDX))
        if add_sos_eos:
            ids.append(EOS_IDX)
        return ids

    def decode(self, ids: List[int], remove_special: bool = True) -> List[str]:
        tokens = []
        for i in ids:
            if remove_special and i in (PAD_IDX, SOS_IDX, EOS_IDX):
                if i == EOS_IDX:
                    break
                continue
            tokens.append(self.idx2word.get(i, UNK_TOKEN))
        return tokens

    def __len__(self):
        return self.n_words

def normalize_english(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKC", text.strip())
    text = text.lower()
    # Normalize quotes, dashes
    text = re.sub(r"[''ʼ`]", "'", text)
    text = re.sub(r'[""“”]', '"', text)
    text = re.sub(r"[—–]", "-", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def normalize_hindi(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKC", text.strip())
    # Separate purna viram (danda) and double danda
    text = re.sub(r"([।॥])", r" \1 ", text)
    text = re.sub(r"[''ʼ`]", "'", text)
    text = re.sub(r'[""“”]', '"', text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def tokenize_english(text: str) -> List[str]:
    # Match alphanumeric words or individual punctuation marks
    tokens = re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?|[^\w\s]", text)
    return tokens

def tokenize_hindi(text: str) -> List[str]:
    # Match Devanagari words, English/numeric words, or single punctuation/danda
    tokens = re.findall(r"[\u0900-\u097F]+|[A-Za-z0-9]+|[^\w\s]", text)
    return tokens

class TranslationDataset(Dataset):
    def __init__(self, pairs: List[Tuple[List[str], List[str]]], src_vocab: Vocabulary, tgt_vocab: Vocabulary):
        self.pairs = pairs
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

        self.encoded_pairs = []
        for src_tokens, tgt_tokens in pairs:
            src_ids = src_vocab.encode(src_tokens, add_sos_eos=True)
            tgt_ids = tgt_vocab.encode(tgt_tokens, add_sos_eos=True)
            self.encoded_pairs.append((src_ids, tgt_ids, src_tokens, tgt_tokens))

    def __len__(self):
        return len(self.encoded_pairs)

    def __getitem__(self, idx):
        src_ids, tgt_ids, src_tokens, tgt_tokens = self.encoded_pairs[idx]
        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long), src_tokens, tgt_tokens

def collate_fn(batch):
    src_tensors, tgt_tensors, src_tokens_list, tgt_tokens_list = zip(*batch)
    
    src_lens = [len(s) for s in src_tensors]
    tgt_lens = [len(t) for t in tgt_tensors]
    
    max_src_len = max(src_lens)
    max_tgt_len = max(tgt_lens)
    
    batch_size = len(batch)
    padded_src = torch.full((batch_size, max_src_len), PAD_IDX, dtype=torch.long)
    padded_tgt = torch.full((batch_size, max_tgt_len), PAD_IDX, dtype=torch.long)
    
    for i in range(batch_size):
        padded_src[i, :len(src_tensors[i])] = src_tensors[i]
        padded_tgt[i, :len(tgt_tensors[i])] = tgt_tensors[i]
        
    return padded_src, torch.tensor(src_lens, dtype=torch.long), padded_tgt, torch.tensor(tgt_lens, dtype=torch.long), src_tokens_list, tgt_tokens_list

def prepare_data(
    train_size: int = 25000,
    max_len: int = 30,
    min_freq: int = 2,
    cache_dir: str = "data"
) -> Dict:
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"preprocessed_{train_size}_{max_len}.pkl")
    stats_file = os.path.join(cache_dir, "corpus_stats.json")

    if os.path.exists(cache_file) and os.path.exists(stats_file):
        print(f"Loading cached preprocessed dataset from {cache_file}...")
        with open(cache_file, "rb") as f:
            data = pickle.load(f)
        return data

    print("Loading official IIT Bombay English-Hindi dataset from cache...")
    raw_ds = load_dataset("cfilt/iitb-english-hindi")

    def clean_and_tokenize_split(split, max_samples: Optional[int] = None, filter_len: bool = True):
        cleaned_pairs = []
        count = 0
        for item in split:
            translation = item.get("translation", {})
            en_raw = translation.get("en", "")
            hi_raw = translation.get("hi", "")

            if not en_raw or not hi_raw:
                continue

            en_norm = normalize_english(en_raw)
            hi_norm = normalize_hindi(hi_raw)

            if not en_norm or not hi_norm:
                continue

            en_tokens = tokenize_english(en_norm)
            hi_tokens = tokenize_hindi(hi_norm)

            if filter_len:
                if len(en_tokens) < 2 or len(en_tokens) > max_len:
                    continue
                if len(hi_tokens) < 2 or len(hi_tokens) > max_len:
                    continue

            cleaned_pairs.append((en_tokens, hi_tokens))
            count += 1
            if max_samples and count >= max_samples:
                break
        return cleaned_pairs

    print(f"Cleaning training split (targeting {train_size} pairs)...")
    train_pairs = clean_and_tokenize_split(raw_ds["train"], max_samples=train_size, filter_len=True)

    print("Cleaning validation split...")
    val_pairs = clean_and_tokenize_split(raw_ds["validation"], max_samples=520, filter_len=False)

    print("Cleaning test split...")
    test_pairs = clean_and_tokenize_split(raw_ds["test"], max_samples=2507, filter_len=False)

    print("Building English and Hindi vocabularies from training data...")
    en_vocab = Vocabulary("English", min_freq=min_freq)
    hi_vocab = Vocabulary("Hindi", min_freq=min_freq)

    for en_toks, hi_toks in train_pairs:
        en_vocab.add_sentence(en_toks)
        hi_vocab.add_sentence(hi_toks)

    en_vocab.build_vocab()
    hi_vocab.build_vocab()

    # Calculate statistics
    # Max lengths include <SOS> and <EOS>
    max_en_train_len = max(len(en) + 2 for en, _ in train_pairs) if train_pairs else 0
    max_hi_train_len = max(len(hi) + 2 for _, hi in train_pairs) if train_pairs else 0

    stats = {
        "num_training_pairs": len(train_pairs),
        "num_validation_pairs": len(val_pairs),
        "num_test_pairs": len(test_pairs),
        "english_vocab_size": len(en_vocab),
        "hindi_vocab_size": len(hi_vocab),
        "max_english_sentence_length": max_en_train_len,
        "max_hindi_sentence_length": max_hi_train_len,
        "min_freq": min_freq,
        "max_token_len_filter": max_len
    }

    print("\n--- IIT Bombay Corpus Statistics ---")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print("------------------------------------\n")

    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=4)

    data = {
        "train_pairs": train_pairs,
        "val_pairs": val_pairs,
        "test_pairs": test_pairs,
        "src_vocab": en_vocab,
        "tgt_vocab": hi_vocab,
        "stats": stats
    }

    with open(cache_file, "wb") as f:
        pickle.dump(data, f)
    print(f"Saved preprocessed data to {cache_file} and stats to {stats_file}.")

    return data

if __name__ == "__main__":
    prepare_data(train_size=25000, max_len=30, min_freq=2)
