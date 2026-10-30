"""Offline ablation B: RoBERTa mean-pool, max_len=128 (seed 42, full train, 5 epochs).
Usage:  python ablation_b_offline.py --data ./amazon_data --model ./roberta-base [--epochs 5] [--workers 4]
  --data  folder with train.jsonl, validation.jsonl, test.jsonl (SetFit/amazon_reviews_multi_en)
  --model folder with roberta-base files (config.json, model.safetensors, vocab.json, merges.txt, tokenizer.json ...)
Writes results to ./outputs/
"""
import os, argparse, random, time, json
os.environ['HF_HUB_OFFLINE'] = '1'; os.environ['HF_DATASETS_OFFLINE'] = '1'; os.environ['TRANSFORMERS_OFFLINE'] = '1'
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import RobertaModel, RobertaTokenizerFast, DataCollatorWithPadding, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score

ap = argparse.ArgumentParser()
ap.add_argument('--data', required=True); ap.add_argument('--model', required=True)
ap.add_argument('--epochs', type=int, default=5); ap.add_argument('--workers', type=int, default=4)
ap.add_argument('--batch', type=int, default=32); ap.add_argument('--accum', type=int, default=1)
args = ap.parse_args()

NUM_CLASSES = 5
DEVICE  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_AMP = torch.cuda.is_available()
RB_EPOCHS = args.epochs
ROBERTA_MODEL = args.model          # local folder
os.makedirs('outputs', exist_ok=True)
print('Device:', DEVICE, torch.cuda.get_device_name(0) if USE_AMP else '')

dataset = load_dataset('json', data_files={
    'train': os.path.join(args.data, 'train.jsonl'),
    'validation': os.path.join(args.data, 'validation.jsonl'),
    'test': os.path.join(args.data, 'test.jsonl')})
train_ds, val_ds, test_ds = dataset['train'], dataset['validation'], dataset['test']
print(f'Train: {len(train_ds):,} | Val: {len(val_ds):,} | Test: {len(test_ds):,}')
assert len(train_ds[0]['text'].strip()) > 0
print(train_ds[0]['text'][:150])

tokenizer = RobertaTokenizerFast.from_pretrained(ROBERTA_MODEL)
collator  = DataCollatorWithPadding(tokenizer, return_tensors='pt')

def tokenize_split(ds, max_len):
    drop = [c for c in ds.column_names if c != 'label']
    out = ds.map(lambda ex: tokenizer([t or '' for t in ex['text']], max_length=max_len, truncation=True, padding=False),
                 batched=True, batch_size=1000, remove_columns=drop)
    out.set_format('torch'); return out
rb_train_128 = tokenize_split(train_ds, 128)
rb_val_128   = tokenize_split(val_ds, 128)
rb_test_128  = tokenize_split(test_ds, 128)

# ── Model ──────────────────────────────────────────────────────────────────────
class RoBERTaClassifier(nn.Module):
    """
    RoBERTa-base with masked mean pooling over the last hidden layer.
    pooling='mean': masked mean of all non-PAD token embeddings (default, required by task)
    pooling='cls' : first-token <s> embedding (ablation)
    """
    def __init__(self, model_name='roberta-base', num_classes=5, dropout=0.1, pooling='mean'):
        super().__init__()
        assert pooling in ('mean', 'cls')
        self.roberta    = RobertaModel.from_pretrained(model_name)
        self.pooling    = pooling
        self.drop       = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.roberta.config.hidden_size, num_classes)

    def mean_pool(self, last_hidden, attention_mask):
        mask  = attention_mask.unsqueeze(-1).float()
        return (last_hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

    def forward(self, input_ids, attention_mask):
        lh     = self.roberta(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        pooled = self.mean_pool(lh, attention_mask) if self.pooling == 'mean' else lh[:, 0]
        return self.classifier(self.drop(pooled))


# ── DataLoaders ────────────────────────────────────────────────────────────────
def make_rb_loaders(tr, va, te, batch_size=16):
    kw = dict(collate_fn=collator, num_workers=args.workers, pin_memory=True)
    return (
        DataLoader(tr, batch_size, shuffle=True,  **kw),
        DataLoader(va, batch_size, shuffle=False, **kw),
        DataLoader(te, batch_size, shuffle=False, **kw),
    )


# ── Evaluation ─────────────────────────────────────────────────────────────────
def eval_roberta(model, loader):
    model.eval()
    preds, trues, total_loss = [], [], 0.0
    crit = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch in loader:
            lk     = 'labels' if 'labels' in batch else 'label'
            labels = batch.pop(lk).to(DEVICE)
            ids    = batch['input_ids'].to(DEVICE)
            mask   = batch['attention_mask'].to(DEVICE)
            logits = model(ids, mask)
            total_loss += crit(logits, labels).item() * len(labels)
            preds.extend(logits.argmax(-1).cpu().numpy())
            trues.extend(labels.cpu().numpy())
    preds, trues = np.array(preds), np.array(trues)
    return (total_loss / len(trues),
            accuracy_score(trues, preds),
            f1_score(trues, preds, average='macro', zero_division=0),
            float(np.mean(np.abs(preds - trues))),
            preds, trues)


# ── Training function ──────────────────────────────────────────────────────────
def train_roberta(seed, pooling='mean', max_len=256,
                  lr=2e-5, weight_decay=0.01, batch_size=16, grad_accum=2,
                  max_epochs=RB_EPOCHS, patience=2, warmup_ratio=0.06, dropout=0.1, tag=''):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

    tr, va, te = (rb_train_128, rb_val_128, rb_test_128) if max_len == 128 \
                 else (rb_train_256, rb_val_256, rb_test_256)
    train_ldr, val_ldr, test_ldr = make_rb_loaders(tr, va, te, batch_size)

    model = RoBERTaClassifier(ROBERTA_MODEL, NUM_CLASSES, dropout, pooling).to(DEVICE)

    no_decay    = ['bias', 'LayerNorm.weight']
    optim_grps  = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0},
    ]
    optim       = torch.optim.AdamW(optim_grps, lr=lr)
    total_steps = (len(train_ldr) // grad_accum) * max_epochs
    sched       = get_linear_schedule_with_warmup(optim, int(total_steps * warmup_ratio), total_steps)
    crit        = nn.CrossEntropyLoss()
    scaler      = torch.cuda.amp.GradScaler() if USE_AMP else None
    label       = tag or f'{pooling}_len{max_len}_s{seed}'

    best_mf1, best_state, no_imp = -1.0, None, 0
    history = dict(train_loss=[], val_loss=[], val_acc=[], val_mf1=[])

    for epoch in range(1, max_epochs + 1):
        model.train(); ep_loss = 0.0; t0 = time.time(); optim.zero_grad()
        for step, batch in enumerate(train_ldr):
            lk     = 'labels' if 'labels' in batch else 'label'
            labels = batch.pop(lk).to(DEVICE)
            ids    = batch['input_ids'].to(DEVICE)
            mask   = batch['attention_mask'].to(DEVICE)

            if USE_AMP:
                with torch.cuda.amp.autocast():
                    loss = crit(model(ids, mask), labels) / grad_accum
                scaler.scale(loss).backward()
            else:
                loss = crit(model(ids, mask), labels) / grad_accum
                loss.backward()

            ep_loss += loss.item() * grad_accum * len(labels)
            if (step + 1) % grad_accum == 0:
                if USE_AMP:
                    scaler.unscale_(optim)
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optim); scaler.update()
                else:
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optim.step()
                sched.step(); optim.zero_grad()

        tr_loss = ep_loss / len(train_ldr.dataset)
        vl, va, vmf1, vmae, _, _ = eval_roberta(model, val_ldr)
        for k, v in zip(['train_loss','val_loss','val_acc','val_mf1'], [tr_loss, vl, va, vmf1]):
            history[k].append(v)
        print(f'[{label}] Ep{epoch:02d} [{time.time()-t0:.0f}s] '
              f'TrLoss={tr_loss:.4f} ValLoss={vl:.4f} ValAcc={va:.4f} ValMF1={vmf1:.4f} ValMAE={vmae:.4f}')

        if vmf1 > best_mf1:
            best_mf1  = vmf1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_imp    = 0
            torch.save(best_state, f'outputs/roberta_best_{label}.pt')
        else:
            no_imp += 1
            if no_imp >= patience:
                print(f'  Early stopping at epoch {epoch}'); break

    model.load_state_dict(best_state)
    _, acc, mf1, mae, preds, trues = eval_roberta(model, test_ldr)
    print(f'[{label}] TEST Acc={acc:.4f} MF1={mf1:.4f} MAE={mae:.4f}')
    return history, acc, mf1, mae, preds, trues, model
if __name__ == '__main__':
    h, acc, mf1, mae, preds, trues, _m = train_roberta(42, pooling='mean', max_len=128, tag='mean128_s42',
                                                       batch_size=args.batch, grad_accum=args.accum)
    res = dict(acc=float(acc), mf1=float(mf1), mae=float(mae), history=h)
    json.dump(res, open('outputs/ablation_b_mean128_s42.json', 'w'), indent=2)
    np.save('outputs/ablation_b_preds.npy', preds); np.save('outputs/ablation_b_trues.npy', trues)
    print(json.dumps(res, indent=2))
