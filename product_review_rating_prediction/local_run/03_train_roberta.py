"""Train a RoBERTa classifier variant for one seed.

Variants:
  main        mean pooling, max_len=256   (required main model; run seeds 42 AND 123)
  ablation_b  mean pooling, max_len=128   (chosen ablation; run seed 42 only)
  ablation_a  CLS pooling,  max_len=256   (the other ablation option; not required, kept for completeness)

Usage:
  python 03_train_roberta.py --variant main --seed 42
  python 03_train_roberta.py --variant main --seed 123
  python 03_train_roberta.py --variant ablation_b --seed 42

Split across two desktops: e.g. run `main --seed 42` here and `main --seed 123`
on the other machine (copy DATA_DIR there first), then copy both machines'
RESULTS_DIR metrics/checkpoints back together before running 04_analyze.py.
"""
import common  # noqa: F401
import argparse, random, time, json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_from_disk
from transformers import RobertaModel, RobertaTokenizerFast, DataCollatorWithPadding, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score

from common import DATA_DIR, RESULTS_DIR, NUM_CLASSES, ROBERTA_MODEL

ap = argparse.ArgumentParser()
ap.add_argument('--variant', choices=['main', 'ablation_b', 'ablation_a'], required=True)
ap.add_argument('--seed', type=int, required=True)
ap.add_argument('--lr', type=float, default=2e-5)
ap.add_argument('--weight_decay', type=float, default=0.01)
ap.add_argument('--batch_size', type=int, default=64)
ap.add_argument('--grad_accum', type=int, default=1)
ap.add_argument('--max_epochs', type=int, default=3)
ap.add_argument('--patience', type=int, default=2)
ap.add_argument('--warmup_ratio', type=float, default=0.06)
ap.add_argument('--dropout', type=float, default=0.1)
ap.add_argument('--workers', type=int, default=8)
args = ap.parse_args()

VARIANT_CFG = {
    'main':       dict(pooling='mean', max_len=256),
    'ablation_b': dict(pooling='mean', max_len=128),
    'ablation_a': dict(pooling='cls',  max_len=256),
}
cfg = VARIANT_CFG[args.variant]
label = f'{args.variant}_s{args.seed}'

DEVICE  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_AMP = torch.cuda.is_available()
print(f'[{label}] Device:', DEVICE, torch.cuda.get_device_name(0) if USE_AMP else '')
print(f'[{label}] pooling={cfg["pooling"]} max_len={cfg["max_len"]}')

tag = 'roberta_256' if cfg['max_len'] == 256 else 'roberta_128'
train_ds = load_from_disk(f'{DATA_DIR}/{tag}/train')
val_ds   = load_from_disk(f'{DATA_DIR}/{tag}/validation')
test_ds  = load_from_disk(f'{DATA_DIR}/{tag}/test')
train_ds.set_format('torch'); val_ds.set_format('torch'); test_ds.set_format('torch')
print(f'train={len(train_ds):,} val={len(val_ds):,} test={len(test_ds):,}')

tokenizer = RobertaTokenizerFast.from_pretrained(ROBERTA_MODEL)
collator  = DataCollatorWithPadding(tokenizer, return_tensors='pt')


class RoBERTaClassifier(nn.Module):
    def __init__(self, model_name, num_classes, dropout, pooling):
        super().__init__()
        assert pooling in ('mean', 'cls')
        self.roberta    = RobertaModel.from_pretrained(model_name)
        self.pooling    = pooling
        self.drop       = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.roberta.config.hidden_size, num_classes)

    def mean_pool(self, last_hidden, attention_mask):
        mask = attention_mask.unsqueeze(-1).float()
        return (last_hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

    def forward(self, input_ids, attention_mask):
        lh = self.roberta(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        pooled = self.mean_pool(lh, attention_mask) if self.pooling == 'mean' else lh[:, 0]
        return self.classifier(self.drop(pooled))


def make_loaders(batch_size):
    kw = dict(collate_fn=collator, num_workers=args.workers, pin_memory=True)
    return (
        DataLoader(train_ds, batch_size, shuffle=True,  **kw),
        DataLoader(val_ds,   batch_size, shuffle=False, **kw),
        DataLoader(test_ds,  batch_size, shuffle=False, **kw),
    )


def eval_model(model, loader):
    model.eval()
    preds, trues, total_loss = [], [], 0.0
    crit = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch in loader:
            lk = 'labels' if 'labels' in batch else 'label'
            labels = batch.pop(lk).to(DEVICE)
            ids  = batch['input_ids'].to(DEVICE)
            mask = batch['attention_mask'].to(DEVICE)
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


def main():
    seed = args.seed
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

    train_ldr, val_ldr, test_ldr = make_loaders(args.batch_size)
    model = RoBERTaClassifier(ROBERTA_MODEL, NUM_CLASSES, args.dropout, cfg['pooling']).to(DEVICE)

    no_decay   = ['bias', 'LayerNorm.weight']
    optim_grps = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0},
    ]
    optim = torch.optim.AdamW(optim_grps, lr=args.lr)
    total_steps = (len(train_ldr) // args.grad_accum) * args.max_epochs
    sched  = get_linear_schedule_with_warmup(optim, int(total_steps * args.warmup_ratio), total_steps)
    crit   = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler() if USE_AMP else None

    best_mf1, best_state, no_imp = -1.0, None, 0
    history = dict(train_loss=[], val_loss=[], val_acc=[], val_mf1=[])
    ckpt_path = f'{RESULTS_DIR}/roberta_best_{label}.pt'

    for epoch in range(1, args.max_epochs + 1):
        model.train(); ep_loss = 0.0; t0 = time.time(); optim.zero_grad()
        for step, batch in enumerate(train_ldr):
            lk = 'labels' if 'labels' in batch else 'label'
            labels = batch.pop(lk).to(DEVICE)
            ids  = batch['input_ids'].to(DEVICE)
            mask = batch['attention_mask'].to(DEVICE)

            if USE_AMP:
                with torch.cuda.amp.autocast():
                    loss = crit(model(ids, mask), labels) / args.grad_accum
                scaler.scale(loss).backward()
            else:
                loss = crit(model(ids, mask), labels) / args.grad_accum
                loss.backward()

            ep_loss += loss.item() * args.grad_accum * len(labels)
            if (step + 1) % args.grad_accum == 0:
                if USE_AMP:
                    scaler.unscale_(optim)
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optim); scaler.update()
                else:
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optim.step()
                sched.step(); optim.zero_grad()

        tr_loss = ep_loss / len(train_ldr.dataset)
        vl, va, vmf1, vmae, _, _ = eval_model(model, val_ldr)
        for k, v in zip(['train_loss', 'val_loss', 'val_acc', 'val_mf1'], [tr_loss, vl, va, vmf1]):
            history[k].append(v)
        print(f'[{label}] Ep{epoch:02d} [{time.time()-t0:.0f}s] '
              f'TrLoss={tr_loss:.4f} ValLoss={vl:.4f} ValAcc={va:.4f} ValMF1={vmf1:.4f} ValMAE={vmae:.4f}')

        if vmf1 > best_mf1:
            best_mf1 = vmf1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_imp = 0
            torch.save(best_state, ckpt_path)
        else:
            no_imp += 1
            if no_imp >= args.patience:
                print(f'  Early stopping at epoch {epoch}'); break

    model.load_state_dict(best_state)
    _, acc, mf1, mae, preds, trues = eval_model(model, test_ldr)
    print(f'[{label}] TEST Acc={acc:.4f} MF1={mf1:.4f} MAE={mae:.4f}')

    result = dict(variant=args.variant, seed=seed, pooling=cfg['pooling'], max_len=cfg['max_len'],
                  history=history, acc=float(acc), mf1=float(mf1), mae=float(mae))
    json.dump(result, open(f'{RESULTS_DIR}/roberta_metrics_{label}.json', 'w'), indent=2)
    np.save(f'{RESULTS_DIR}/roberta_preds_{label}.npy', preds)
    np.save(f'{RESULTS_DIR}/roberta_trues_{label}.npy', trues)
    print('Saved to', RESULTS_DIR)


if __name__ == '__main__':
    main()
