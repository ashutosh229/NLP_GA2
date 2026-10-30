"""Save per-class precision/recall/F1 (required by task.txt Sec.5 Quantitative)
for both models' best seed, as a structured file (not just console output).
"""
import common  # noqa: F401
import json
import numpy as np
from sklearn.metrics import classification_report

from common import SEEDS, RESULTS_DIR

bigru_mf1   = {s: json.load(open(f'{RESULTS_DIR}/bigru_metrics_seed{s}.json'))['mf1'] for s in SEEDS}
roberta_mf1 = {s: json.load(open(f'{RESULTS_DIR}/roberta_metrics_main_s{s}.json'))['mf1'] for s in SEEDS}
best_bg = max(SEEDS, key=lambda s: bigru_mf1[s])
best_rb = max(SEEDS, key=lambda s: roberta_mf1[s])
print(f'Best BiGRU seed: {best_bg} (MF1={bigru_mf1[best_bg]:.4f})')
print(f'Best RoBERTa seed: {best_rb} (MF1={roberta_mf1[best_rb]:.4f})')

bg_preds = np.load(f'{RESULTS_DIR}/bigru_preds_seed{best_bg}.npy')
bg_trues = np.load(f'{RESULTS_DIR}/bigru_trues_seed{best_bg}.npy')
rb_preds = np.load(f'{RESULTS_DIR}/roberta_preds_main_s{best_rb}.npy')
rb_trues = np.load(f'{RESULTS_DIR}/roberta_trues_main_s{best_rb}.npy')

star_names = [f'{i+1}-star' for i in range(5)]
bg_report = classification_report(bg_trues, bg_preds, target_names=star_names, output_dict=True, zero_division=0)
rb_report = classification_report(rb_trues, rb_preds, target_names=star_names, output_dict=True, zero_division=0)

out = {
    'bigru':   {'best_seed': best_bg, 'report': bg_report},
    'roberta': {'best_seed': best_rb, 'report': rb_report},
}
json.dump(out, open(f'{RESULTS_DIR}/per_class_classification_report.json', 'w'), indent=2)

print('\nPer-class F1 (BiGRU vs RoBERTa):')
print(f'{"Class":<10} {"BiGRU F1":>10} {"RoBERTa F1":>12}')
for name in star_names:
    print(f'{name:<10} {bg_report[name]["f1-score"]:>10.3f} {rb_report[name]["f1-score"]:>12.3f}')
print('Saved to', f'{RESULTS_DIR}/per_class_classification_report.json')
