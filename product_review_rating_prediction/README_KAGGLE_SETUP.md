# Product Review Rating Prediction — Kaggle Setup Guide

## Quick Overview

You have two execution choices:

### 🌟 Option 1: All-In-One (Recommended — No Output Carryforward Needed)
| Notebook | Purpose |
|---|---|
| [`kaggle_main.ipynb`](file:///c:/Users/kishor/Pictures/nlp/nlpga2/NLP_GA2/product_review_rating_prediction/kaggle_main.ipynb) | Complete end-to-end pipeline in a single notebook: data loading, preprocessing, BiGRU training, RoBERTa training, ablations, comparisons, and visualizations. Just upload to Kaggle and click **Run All**. |

---

### Option 2: Modular Pipeline (3 Notebooks)
If you prefer running in separate stages:
| Notebook | Purpose |
|---|---|
| [`notebook1_preprocessing.ipynb`](file:///c:/Users/kishor/Pictures/nlp/nlpga2/NLP_GA2/product_review_rating_prediction/notebook1_preprocessing.ipynb) | Data loading, classic NLP preprocessing (vocab + embeddings), RoBERTa tokenization |
| [`notebook2_bigru_baseline.ipynb`](file:///c:/Users/kishor/Pictures/nlp/nlpga2/NLP_GA2/product_review_rating_prediction/notebook2_bigru_baseline.ipynb) | BiGRU + Additive Attention training and evaluation |
| [`notebook3_roberta.ipynb`](file:///c:/Users/kishor/Pictures/nlp/nlpga2/NLP_GA2/product_review_rating_prediction/notebook3_roberta.ipynb) | RoBERTa fine-tuning, ablation study, comparative analysis |

---

## Kaggle Setup Instructions

### 1. Create a New Kaggle Notebook

Go to [kaggle.com/code](https://kaggle.com/code) → **+ New Notebook**

### 2. Enable GPU Accelerator

**Settings (right panel)** → **Accelerator** → Select **GPU T4 x2** or **P100**

### 3. Add Pre-trained Word Vectors as Input (Optional but Recommended)

You can attach either FastText (preferred by task) or GloVe directly from Kaggle so you don't have to wait for downloading:

| Embedding | Direct Kaggle Link | Search Keyword / Slug | Mount Path in Notebook |
|---|---|---|---|
| **FastText Crawl 300d 2M** *(Recommended)* | [yekenot/fasttext-crawl-300d-2m](https://www.kaggle.com/datasets/yekenot/fasttext-crawl-300d-2m) | `fasttext-crawl-300d-2m` | `/kaggle/input/fasttext-crawl-300d-2m/crawl-300d-2M.vec` |
| **GloVe 840B 300d** *(Alternative fallback)* | [takuok/glove840b300dtxt](https://www.kaggle.com/datasets/takuok/glove840b300dtxt) | `glove840b300dtxt` | `/kaggle/input/glove840b300dtxt/glove.840B.300d.txt` |

> 💡 **Notice:** Even if you don't attach anything, the notebook contains an **automatic fallback** that downloads `crawl-300d-2M.vec.zip` directly from Meta AI (`dl.fbaipublicfiles.com`). Attaching the Kaggle dataset simply saves ~3-5 minutes of download time.

> **Note on Dataset:** The review classification dataset `SetFit/amazon_reviews_multi_en` does **not** need to be added as Kaggle input; it downloads automatically from Hugging Face via `load_dataset()`.

### 4. Run Order

Run the three notebooks **in order**. Outputs from Notebook 1 (`processed/` folder) are required by Notebooks 2 and 3.

**Notebook 1 → Notebook 2 → Notebook 3**

Use Kaggle's "Save & Run All" to run each sequentially.

---

## Architecture Summary

### Baseline — BiGRU + Additive Attention

```
Input tokens (max 200)
    → Embedding Layer (fastText 300d, frozen epoch 1, unfrozen epoch 2+)
    → Bidirectional GRU (2 layers, hidden=256, dropout=0.3)
    → Additive Attention Pooling (Bahdanau-style)
    → Dropout (0.3)
    → Linear (512 → 5)
```

### RoBERTa Model

```
Input text (max 256 tokens, byte-level BPE)
    → RobertaModel (roberta-base, 125M params)
    → Masked Mean Pooling over last hidden layer
    → Dropout (0.1)
    → Linear (768 → 5)
```

---

## Training Configuration

### BiGRU Baseline
| Hyperparameter | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | 1e-3 (embed: 1e-4 after unfreeze) |
| Batch size | 128 |
| Gradient clip | 1.0 |
| Early stopping | patience=3 on Val Macro F1 |
| Seeds | 42, 123 |

### RoBERTa
| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 2e-5 |
| Weight decay | 0.01 |
| Batch size | 16 (effective 32 with grad_accum=2) |
| Warmup | 6% linear warmup + linear decay |
| Mixed precision | fp16 (AMP) |
| Early stopping | patience=2 on Val Macro F1 |
| Seeds | 42, 123 |

---

## Ablation Study

**Ablation**: Mean Pooling vs. First-Token (CLS) Pooling *(primary)*  
**Bonus Ablation**: Max Length 128 vs. 256 tokens

Both run in Notebook 3 with seed=42.

---

## Expected Output Files

After running all notebooks, you'll have:

```
bigru_training_curves.png
bigru_confusion_matrix.png
bigru_results.json
bigru_test_preds.npy
bigru_test_trues.npy

roberta_training_curves.png
roberta_confusion_matrix.png
roberta_token_importance.png
roberta_results.json
roberta_test_preds.npy
roberta_test_trues.npy

comparison_confusion_matrices.png
processed/label_dist.png
```

---

## Estimated Runtimes (Kaggle GPU T4)

| Stage | Estimated Time |
|---|---|
| Notebook 1: Preprocessing | ~90 minutes (200K tokenizations) |
| Notebook 2: BiGRU (2 seeds × ~10 epochs) | ~60 minutes |
| Notebook 3: RoBERTa (2 seeds + 2 ablations) | ~4–6 hours |

> **Tip**: Run Notebooks 1 and 2 first, save the outputs, then start Notebook 3 in a separate session to avoid the 9-hour limit.

---

## Notes on the Dataset

The `SetFit/amazon_reviews_multi_en` dataset has:
- **Train**: 200,000 reviews
- **Validation**: 5,000 reviews  
- **Test**: 5,000 reviews

Labels are 0–4 (mapped from 1–5 stars). The task spec says 5 classes; labels in this dataset start from 0.

Each sample has `review_title` and `review_body` — both are concatenated as the input text.

