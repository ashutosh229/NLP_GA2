"""Combine all trained results: comparison tables, confusion matrices, training
curves, ordinal error analysis, qualitative examples, and RoBERTa token-importance
saliency plot. Run after 02_train_bigru.py (both seeds) and 03_train_roberta.py
(main both seeds + ablation_b seed 42).
"""
import common  # noqa: F401
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from datasets import load_dataset
from sklearn.metrics import confusion_matrix, classification_report
from transformers import RobertaModel, RobertaTokenizerFast

from common import DATA_DIR, RESULTS_DIR, SEEDS, DATASET_NAME, ROBERTA_MODEL

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_text(sample):
    if sample.get('text'):
        return sample['text'].strip()
    return ((sample.get('review_title') or '') + ' ' + (sample.get('review_body') or '')).strip()


# ── Load all metrics ─────────────────────────────────────────────────────────
bigru_results = {s: json.load(open(f'{RESULTS_DIR}/bigru_metrics_seed{s}.json')) for s in SEEDS}
roberta_main  = {s: json.load(open(f'{RESULTS_DIR}/roberta_metrics_main_s{s}.json')) for s in SEEDS}
abl_b = json.load(open(f'{RESULTS_DIR}/roberta_metrics_ablation_b_s42.json'))

bg_accs = [bigru_results[s]['acc'] for s in SEEDS]
bg_mf1s = [bigru_results[s]['mf1'] for s in SEEDS]
bg_maes = [bigru_results[s]['mae'] for s in SEEDS]
rb_accs = [roberta_main[s]['acc'] for s in SEEDS]
rb_mf1s = [roberta_main[s]['mf1'] for s in SEEDS]
rb_maes = [roberta_main[s]['mae'] for s in SEEDS]

print(f'\n=== BiGRU Final (mean ± std over {len(SEEDS)} seeds) ===')
print(f'Accuracy : {np.mean(bg_accs):.4f} ± {np.std(bg_accs):.4f}')
print(f'Macro F1 : {np.mean(bg_mf1s):.4f} ± {np.std(bg_mf1s):.4f}')
print(f'MAE      : {np.mean(bg_maes):.4f} ± {np.std(bg_maes):.4f}')

print(f'\n=== RoBERTa Main (mean pooling, 256 tok, mean ± std over {len(SEEDS)} seeds) ===')
print(f'Accuracy : {np.mean(rb_accs):.4f} ± {np.std(rb_accs):.4f}')
print(f'Macro F1 : {np.mean(rb_mf1s):.4f} ± {np.std(rb_mf1s):.4f}')
print(f'MAE      : {np.mean(rb_maes):.4f} ± {np.std(rb_maes):.4f}')

print(f'\n=== Ablation B: Mean Pooling, max_len=128, seed=42 ===')
print(f'Accuracy : {abl_b["acc"]:.4f}')
print(f'Macro F1 : {abl_b["mf1"]:.4f}')
print(f'MAE      : {abl_b["mae"]:.4f}')

print(f'\n{"─"*62}')
print(f'{"Config":<32} {"Acc":>8} {"MF1":>8} {"MAE":>8}')
print(f'{"─"*62}')
print(f'{"RoBERTa mean-pool, len=256 (main)":<32} {np.mean(rb_accs):>8.4f} {np.mean(rb_mf1s):>8.4f} {np.mean(rb_maes):>8.4f}')
print(f'{"RoBERTa mean-pool, len=128 (ablation B)":<32} {abl_b["acc"]:>8.4f} {abl_b["mf1"]:>8.4f} {abl_b["mae"]:>8.4f}')
print(f'{"─"*62}')

# ── Training curves ─────────────────────────────────────────────────────────
def plot_curves(history, title, path):
    ep = range(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(ep, history['train_loss'], 'o-', label='Train Loss')
    axes[0].plot(ep, history['val_loss'],   's-', label='Val Loss')
    axes[0].set(xlabel='Epoch', ylabel='Loss', title=f'{title} — Loss'); axes[0].legend(); axes[0].grid(True)
    axes[1].plot(ep, history['val_acc'], 'o-', label='Val Accuracy')
    axes[1].plot(ep, history['val_mf1'], 's-', label='Val Macro F1')
    axes[1].set(xlabel='Epoch', ylabel='Score', title=f'{title} — Val Metrics'); axes[1].legend(); axes[1].grid(True)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close(fig)

best_bg_seed = max(SEEDS, key=lambda s: bigru_results[s]['mf1'])
best_rb_seed = max(SEEDS, key=lambda s: roberta_main[s]['mf1'])
plot_curves(bigru_results[best_bg_seed]['history'], 'BiGRU', f'{RESULTS_DIR}/bigru_training_curves.png')
plot_curves(roberta_main[best_rb_seed]['history'],  'RoBERTa', f'{RESULTS_DIR}/roberta_training_curves.png')
plot_curves(abl_b['history'], 'RoBERTa Ablation B (len=128)', f'{RESULTS_DIR}/roberta_ablation_b_curves.png')

# ── Predictions (best seed per model) ───────────────────────────────────────
bg_preds = np.load(f'{RESULTS_DIR}/bigru_preds_seed{best_bg_seed}.npy')
bg_trues = np.load(f'{RESULTS_DIR}/bigru_trues_seed{best_bg_seed}.npy')
rb_preds = np.load(f'{RESULTS_DIR}/roberta_preds_main_s{best_rb_seed}.npy')
rb_trues = np.load(f'{RESULTS_DIR}/roberta_trues_main_s{best_rb_seed}.npy')

star_names = [f'{i+1}-star' for i in range(5)]
print('\n=== BiGRU Classification Report (best seed) ===')
print(classification_report(bg_trues, bg_preds, target_names=star_names))
print('\n=== RoBERTa Classification Report (best seed) ===')
print(classification_report(rb_trues, rb_preds, target_names=star_names))

# ── Confusion matrices ───────────────────────────────────────────────────────
def norm_cm(preds, trues):
    cm = confusion_matrix(trues, preds)
    return cm.astype(float) / cm.sum(axis=1, keepdims=True)

labels = [f'{i+1}★' for i in range(5)]
bg_cm, rb_cm = norm_cm(bg_preds, bg_trues), norm_cm(rb_preds, rb_trues)
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for ax, cm, title, cmap in zip(axes, [bg_cm, rb_cm],
                                ['BiGRU — Confusion Matrix', 'RoBERTa — Confusion Matrix'],
                                ['Oranges', 'Blues']):
    sns.heatmap(cm, annot=True, fmt='.2f', cmap=cmap, ax=ax, xticklabels=labels, yticklabels=labels)
    ax.set(xlabel='Predicted', ylabel='True', title=title)
plt.tight_layout(); plt.savefig(f'{RESULTS_DIR}/confusion_matrices.png', dpi=150); plt.close(fig)

# ── Ordinal error analysis ───────────────────────────────────────────────────
def ordinal_analysis(preds, trues, name):
    d = np.abs(np.array(preds) - np.array(trues))
    print(f'\nOrdinal Error Analysis — {name}')
    out = {}
    for label_, cond in [('correct_diff0', d == 0), ('adjacent_diff1', d == 1),
                          ('near_miss_diff2', d == 2), ('distant_diff_ge3', d >= 3)]:
        out[label_] = float(cond.mean())
        print(f'  {label_}: {out[label_]:.3f}')
    return out

ordinal_bg = ordinal_analysis(bg_preds, bg_trues, 'BiGRU')
ordinal_rb = ordinal_analysis(rb_preds, rb_trues, 'RoBERTa')

# ── Qualitative examples ─────────────────────────────────────────────────────
dataset = load_dataset(DATASET_NAME)
test_ds = dataset['test']

n = min(len(bg_preds), len(rb_preds), len(test_ds))
bg_p, rb_p, true = np.array(bg_preds[:n]), np.array(rb_preds[:n]), np.array(bg_trues[:n])

bf_rw = np.where((bg_p != true) & (rb_p == true))[0]
rf_bw = np.where((rb_p != true) & (bg_p == true))[0]
print(f'\nBiGRU fails / RoBERTa wins: {len(bf_rw):,}')
print(f'RoBERTa fails / BiGRU wins: {len(rf_bw):,}')

def show_examples(indices, title, k=5):
    rng = np.random.default_rng(0)
    chosen = rng.choice(indices, size=min(k, len(indices)), replace=False) if len(indices) else []
    lines = [f'\n{"="*70}\n{title}\n{"="*70}']
    for i, idx in enumerate(chosen):
        s = test_ds[int(idx)]
        text = get_text(s)
        lines.append(f'\n[{i+1}] True={true[idx]+1}star  BiGRU={bg_p[idx]+1}star  RoBERTa={rb_p[idx]+1}star')
        lines.append(f'Text: {text[:350]}')
    out = '\n'.join(lines)
    print(out)
    return out

ex1 = show_examples(bf_rw, 'BiGRU FAILS -- RoBERTa SUCCEEDS', k=5)
ex2 = show_examples(rf_bw, 'RoBERTa FAILS -- BiGRU SUCCEEDS', k=2)
with open(f'{RESULTS_DIR}/qualitative_examples.txt', 'w') as f:
    f.write(ex1 + '\n' + ex2)

# ── Token-importance saliency (gradient norm) for RoBERTa ──────────────────
tokenizer = RobertaTokenizerFast.from_pretrained(ROBERTA_MODEL)

class RoBERTaClassifier(nn.Module):
    def __init__(self, model_name, num_classes=5, dropout=0.1, pooling='mean'):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained(model_name)
        self.pooling = pooling
        self.drop = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.roberta.config.hidden_size, num_classes)
    def mean_pool(self, last_hidden, attention_mask):
        mask = attention_mask.unsqueeze(-1).float()
        return (last_hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
    def forward(self, input_ids, attention_mask):
        lh = self.roberta(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        pooled = self.mean_pool(lh, attention_mask) if self.pooling == 'mean' else lh[:, 0]
        return self.classifier(self.drop(pooled))

rb_model = RoBERTaClassifier(ROBERTA_MODEL, 5, 0.1, 'mean').to(DEVICE)
rb_model.load_state_dict(torch.load(f'{RESULTS_DIR}/roberta_best_main_s{best_rb_seed}.pt', map_location=DEVICE))
rb_model.eval()

def token_importance(model, text, true_label, path):
    inp = tokenizer(text, return_tensors='pt', max_length=256, truncation=True).to(DEVICE)
    emb = model.roberta.embeddings(inp['input_ids'])
    emb.retain_grad()
    lh = model.roberta(inputs_embeds=emb, attention_mask=inp['attention_mask']).last_hidden_state
    mask = inp['attention_mask'].unsqueeze(-1).float()
    p = (lh * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
    logits = model.classifier(model.drop(p))
    logits[0, true_label].backward()

    imp = emb.grad[0].detach().cpu().norm(dim=-1).numpy()
    tokens = tokenizer.convert_ids_to_tokens(inp['input_ids'][0].cpu().tolist())
    n_ = len(tokens)
    imp = (imp[:n_] - imp[:n_].min()) / (np.ptp(imp[:n_]) + 1e-9)
    pred = logits.argmax().item() + 1

    fig, ax = plt.subplots(figsize=(max(10, n_ // 3), 3))
    ax.bar(range(n_), imp, color='steelblue')
    ax.set_xticks(range(n_)); ax.set_xticklabels(tokens, rotation=90, fontsize=8)
    ax.set(ylabel='Gradient Importance', title=f'Token Importance | True: {true_label+1}star Pred: {pred}star')
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close(fig)

if len(bf_rw) > 0:
    idx = int(bf_rw[0])
    s = test_ds[idx]
    txt = get_text(s)
    print(f'\nVisualising idx={idx} | True={true[idx]+1}star')
    token_importance(rb_model, txt, int(true[idx]), f'{RESULTS_DIR}/token_importance.png')

# ── Final summary JSON ───────────────────────────────────────────────────────
results = {
    'bigru': {
        'acc_mean': float(np.mean(bg_accs)), 'acc_std': float(np.std(bg_accs)),
        'mf1_mean': float(np.mean(bg_mf1s)), 'mf1_std': float(np.std(bg_mf1s)),
        'mae_mean': float(np.mean(bg_maes)), 'mae_std': float(np.std(bg_maes)),
        'ordinal_error': ordinal_bg,
    },
    'roberta_main': {
        'acc_mean': float(np.mean(rb_accs)), 'acc_std': float(np.std(rb_accs)),
        'mf1_mean': float(np.mean(rb_mf1s)), 'mf1_std': float(np.std(rb_mf1s)),
        'mae_mean': float(np.mean(rb_maes)), 'mae_std': float(np.std(rb_maes)),
        'ordinal_error': ordinal_rb,
    },
    'ablation_b_mean128': {'acc': abl_b['acc'], 'mf1': abl_b['mf1'], 'mae': abl_b['mae']},
    'bigru_fails_roberta_wins': int(len(bf_rw)),
    'roberta_fails_bigru_wins': int(len(rf_bw)),
}
json.dump(results, open(f'{RESULTS_DIR}/all_results.json', 'w'), indent=2)
print('\nSaved all_results.json and figures to', RESULTS_DIR)
print(json.dumps(results, indent=2))
