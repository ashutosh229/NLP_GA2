"""Train the BiGRU + additive-attention baseline for one seed.

Usage: python 02_train_bigru.py --seed 42
       python 02_train_bigru.py --seed 123
Run the two seeds on two different machines in parallel if you want to split work
(each only needs DATA_DIR from 01_prepare_data.py, no GPU-to-GPU communication).
"""
import common  # noqa: F401
import argparse, pickle, random, time, json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score

from common import DATA_DIR, RESULTS_DIR, EMBED_DIM

ap = argparse.ArgumentParser()
ap.add_argument('--seed', type=int, required=True)
ap.add_argument('--hidden_dim', type=int, default=256)
ap.add_argument('--lr', type=float, default=1e-3)
ap.add_argument('--batch_size', type=int, default=128)
ap.add_argument('--max_epochs', type=int, default=15)
ap.add_argument('--patience', type=int, default=3)
ap.add_argument('--grad_clip', type=float, default=1.0)
ap.add_argument('--workers', type=int, default=8)
args = ap.parse_args()

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', DEVICE, torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')

vocab_info = pickle.load(open(f'{DATA_DIR}/vocab.pkl', 'rb'))
vocab = vocab_info['vocab']
embedding_matrix = np.load(f'{DATA_DIR}/embedding_matrix.npy')

train_ids = np.load(f'{DATA_DIR}/classic_train_ids.npy')
val_ids   = np.load(f'{DATA_DIR}/classic_val_ids.npy')
test_ids  = np.load(f'{DATA_DIR}/classic_test_ids.npy')
train_lbl = np.load(f'{DATA_DIR}/classic_train_labels.npy')
val_lbl   = np.load(f'{DATA_DIR}/classic_val_labels.npy')
test_lbl  = np.load(f'{DATA_DIR}/classic_test_labels.npy')
print(f'train={train_ids.shape} val={val_ids.shape} test={test_ids.shape}')


class ReviewDataset(Dataset):
    def __init__(self, ids, labels):
        self.ids    = torch.tensor(ids,    dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)
    def __len__(self): return len(self.ids)
    def __getitem__(self, i): return self.ids[i], self.labels[i]


def make_loaders(batch_size):
    kw = dict(num_workers=args.workers, pin_memory=True)
    return (
        DataLoader(ReviewDataset(train_ids, train_lbl), batch_size, shuffle=True,  **kw),
        DataLoader(ReviewDataset(val_ids,   val_lbl),   batch_size, shuffle=False, **kw),
        DataLoader(ReviewDataset(test_ids,  test_lbl),  batch_size, shuffle=False, **kw),
    )


class AdditiveAttention(nn.Module):
    """Bahdanau-style additive attention pooling."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.w = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, h, pad_mask=None):
        scores = self.v(torch.tanh(self.w(h))).squeeze(-1)
        if pad_mask is not None:
            all_pad = pad_mask.all(dim=-1, keepdim=True)
            safe_mask = pad_mask & (~all_pad)
            scores = scores.masked_fill(safe_mask, -1e4)
        weights = scores.softmax(dim=-1)
        context = torch.bmm(weights.unsqueeze(1), h).squeeze(1)
        return context, weights


class BiGRUClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim=256, num_layers=2,
                 num_classes=5, dropout=0.3, emb_weights=None, freeze=True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if emb_weights is not None:
            self.embedding.weight.data.copy_(torch.tensor(emb_weights, dtype=torch.float32))
        self.embedding.weight.requires_grad_(not freeze)

        self.gru = nn.GRU(embed_dim, hidden_dim, num_layers=num_layers,
                           batch_first=True, bidirectional=True,
                           dropout=dropout if num_layers > 1 else 0.0)
        self.attn       = AdditiveAttention(hidden_dim * 2)
        self.drop       = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, ids):
        pad_mask = (ids == 0)
        x = self.drop(self.embedding(ids))
        h, _ = self.gru(x)
        ctx, _ = self.attn(h, pad_mask)
        return self.classifier(self.drop(ctx))

    def unfreeze_embeddings(self):
        self.embedding.weight.requires_grad_(True)


def eval_model(model, loader):
    model.eval()
    preds, trues, total_loss = [], [], 0.0
    crit = nn.CrossEntropyLoss()
    with torch.no_grad():
        for ids, labels in loader:
            ids, labels = ids.to(DEVICE), labels.to(DEVICE)
            logits = model(ids)
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

    train_loader, val_loader, test_loader = make_loaders(args.batch_size)
    model = BiGRUClassifier(len(vocab), EMBED_DIM, hidden_dim=args.hidden_dim,
                             emb_weights=embedding_matrix, freeze=True).to(DEVICE)
    optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, 'max', patience=1, factor=0.5)
    crit  = nn.CrossEntropyLoss()
    best_mf1, best_state, no_imp = -1.0, None, 0
    history = dict(train_loss=[], val_loss=[], val_acc=[], val_mf1=[])
    ckpt_path = f'{RESULTS_DIR}/bigru_best_seed{seed}.pt'

    for epoch in range(1, args.max_epochs + 1):
        if epoch == 2:
            model.unfreeze_embeddings()
            optim.add_param_group({'params': model.embedding.parameters(), 'lr': args.lr * 0.1})
            print('  [embeddings unfrozen]')

        model.train(); ep_loss = 0.0; t0 = time.time()
        for ids, labels in train_loader:
            ids, labels = ids.to(DEVICE), labels.to(DEVICE)
            optim.zero_grad()
            loss = crit(model(ids), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optim.step()
            ep_loss += loss.item() * len(labels)

        tr_loss = ep_loss / len(train_loader.dataset)
        vl, va, vmf1, vmae, _, _ = eval_model(model, val_loader)
        sched.step(vmf1)
        for k, v in zip(['train_loss', 'val_loss', 'val_acc', 'val_mf1'], [tr_loss, vl, va, vmf1]):
            history[k].append(v)
        print(f'[BiGRU s{seed}] Ep{epoch:02d} [{time.time()-t0:.0f}s] '
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
    _, acc, mf1, mae, preds, trues = eval_model(model, test_loader)
    print(f'[BiGRU s{seed}] TEST Acc={acc:.4f} MF1={mf1:.4f} MAE={mae:.4f}')

    result = dict(seed=seed, history=history, acc=float(acc), mf1=float(mf1), mae=float(mae))
    json.dump(result, open(f'{RESULTS_DIR}/bigru_metrics_seed{seed}.json', 'w'), indent=2)
    np.save(f'{RESULTS_DIR}/bigru_preds_seed{seed}.npy', preds)
    np.save(f'{RESULTS_DIR}/bigru_trues_seed{seed}.npy', trues)
    print('Saved to', RESULTS_DIR)


if __name__ == '__main__':
    main()
