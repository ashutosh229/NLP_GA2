# Local GPU training pipeline

Replaces the slow Kaggle run (`kaggle_main.ipynb`) with scripts for this desktop's GPU.
Chosen ablation: **Ablation B — mean pooling, max_len=128** (vs. the main model's
mean pooling, max_len=256). Ablation A (CLS pooling) is not run, per your choice.

Caches/downloads go to `/data3/Aditya/nlpga2_cache` (see `common.py`) to avoid filling
the near-full root disk.

## Single machine

```bash
cd local_run
bash run_all.sh
```

## Splitting across two desktops (optional, for parallelism)

1. Run `python 01_prepare_data.py` **once**, on either machine.
2. `rsync -av /data3/Aditya/nlpga2_cache/data/ <other-host>:/data3/Aditya/nlpga2_cache/data/`
   (copies the vocab/embeddings/tokenized datasets so the other desktop doesn't redo it).
3. Split the five training jobs across the two machines, e.g.:
   - **Machine A**: `02_train_bigru.py --seed 42`, `03_train_roberta.py --variant main --seed 42`
   - **Machine B**: `02_train_bigru.py --seed 123`, `03_train_roberta.py --variant main --seed 123`, `03_train_roberta.py --variant ablation_b --seed 42`
4. Copy every `local_results/*` file (`bigru_metrics_*.json`, `bigru_preds/trues_*.npy`,
   `roberta_metrics_*.json`, `roberta_preds/trues_*.npy`, `roberta_best_*.pt`) from Machine B
   back into Machine A's `RESULTS_DIR` (`/data3/Aditya/nlpga2_cache/results`).
5. Run `python 04_analyze.py` on Machine A once everything is merged.

## Outputs

Under `/data3/Aditya/nlpga2_cache/results/`:
- `bigru_metrics_seed{42,123}.json`, `bigru_best_seed{42,123}.pt`
- `roberta_metrics_main_s{42,123}.json`, `roberta_metrics_ablation_b_s42.json`
- `*_training_curves.png`, `confusion_matrices.png`, `token_importance.png`
- `qualitative_examples.txt`, `all_results.json` (final comparison table)
