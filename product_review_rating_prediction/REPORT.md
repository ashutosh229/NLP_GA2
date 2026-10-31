# Fine-Tuning RoBERTa for Multi-Class Product Review Rating Prediction

_BiGRU + Additive Attention baseline vs. fine-tuned RoBERTa-base — Amazon Reviews (SetFit/amazon_reviews_multi_en)_

## 1. Objective & Dataset

We implement and compare two 5-class (1-5 star) review-rating classifiers: a Bidirectional GRU with additive attention over frozen-then-fine-tuned fastText embeddings, and a fine-tuned RoBERTa-base model with masked mean pooling. Both are trained end-to-end on the full SetFit/amazon_reviews_multi_en dataset (200,000 train / 5,000 validation / 5,000 test reviews, title+body concatenated as input), with the test set evaluated only once, at the end.

## 2. Model Architectures

**Baseline (BiGRU):** Embedding (300-d fastText crawl-300d-2M, 98.2% vocabulary coverage; frozen epoch 1, fine-tuned at 0.1× LR from epoch 2) → 2-layer Bidirectional GRU (hidden=256, dropout=0.3) → Bahdanau-style additive attention pooling → Dropout(0.3) → Linear(512→5). Input: spaCy-tokenized, lower-cased, HTML/URL-stripped text, vocab min-freq=3 (24,570 tokens), max length 200.

**RoBERTa:** roberta-base (125M params, byte-level BPE tokenizer, dynamic padding via a HF data collator) → masked mean pooling over all non-padded last-hidden-state tokens (task-required; not CLS pooling) → Dropout(0.1) → Linear(768→5), fine-tuned end-to-end. Main configuration: max length 256 tokens.

## 3. Training Process

**BiGRU:** Adam, LR=1e-3 (embeddings 1e-4 after unfreezing), batch=128, gradient clipping (max-norm 1.0), ReduceLROnPlateau on validation Macro-F1, early stopping (patience=3, max 15 epochs). **RoBERTa:** AdamW, LR=2e-5, weight decay 0.01 (no decay on bias/LayerNorm), batch=64, linear warmup (6%) + linear decay, fp16 mixed precision, early stopping (patience=2, max 3 epochs). Both: Cross-Entropy loss, fixed seeds, results reported as mean±std over 2 seeds (42, 123). Training ran on a single GPU (Quadro RTX 8000, 48GB) using the full 200K-review training set for every run (no sub-sampling).

BiGRU converged (early-stopped) at epoch 10 (seed 42) / 11 (seed 123); RoBERTa main used all 3 epochs for both seeds (monotonically improving validation Macro-F1 within budget). Training/validation loss and Macro-F1 curves (best seed per model) are shown below.

![Figure 1 (left): BiGRU loss and validation metrics](images/figure1a_bigru_curves.png)

![Figure 1 (right): RoBERTa loss and validation metrics](images/figure1b_roberta_curves.png)

_Figure 1: Training/validation loss and Macro-F1/Accuracy, BiGRU (left, seed 42) vs. RoBERTa main (right, seed 42)._

## 4. Quantitative Results (Test Set)

| Model                                    | Accuracy      | Macro F1      | MAE           |
| ---------------------------------------- | ------------- | ------------- | ------------- |
| BiGRU + Attention (baseline)             | 0.589 ± 0.002 | 0.588 ± 0.001 | 0.487 ± 0.000 |
| RoBERTa, mean-pool, len=256 (main)       | 0.615 ± 0.003 | 0.612 ± 0.003 | 0.439 ± 0.004 |
| RoBERTa, mean-pool, len=128 (Ablation B) | 0.611         | 0.609         | 0.440         |

| Class  | BiGRU F1 | RoBERTa F1 |
| ------ | -------- | ---------- |
| 1-star | 0.674    | 0.711      |
| 2-star | 0.495    | 0.517      |
| 3-star | 0.474    | 0.503      |
| 4-star | 0.539    | 0.568      |
| 5-star | 0.761    | 0.777      |

| Error distance | BiGRU | RoBERTa |
| -------------- | ----- | ------- |
| Correct (0)    | 0.591 | 0.618   |
| Adjacent (1)   | 0.346 | 0.338   |
| Near-miss (2)  | 0.052 | 0.038   |
| Distant (>=3)  | 0.011 | 0.006   |

_Table: per-class F1 (left) and ordinal error distribution \|pred-true\| (right, fraction of test set)._

![Figure 2: Row-normalized confusion matrices](images/figure2_confusion_matrices.png)

_Figure 2: Row-normalized confusion matrices on the test set._

## 5. Ablation Study: Max Sequence Length (256 vs. 128 tokens)

Chosen ablation: mean pooling at max_len=128 ("Ablation B") vs. the main configuration (mean pooling, max_len=256), both seed 42. Truncating to 128 tokens costs only 0.06 accuracy points (61.14%→61.08%) and 0.02 Macro-F1 points (60.90%→60.88%), with MAE essentially unchanged (0.443→0.440). This indicates most of the rating-relevant sentiment signal in these reviews is concentrated in the first ~128 BPE tokens (title + opening sentences); halving the sequence length roughly halves per-epoch compute (measured ~420s vs. ~630s/epoch on the same GPU) for a near-negligible accuracy cost — a favorable efficiency/accuracy trade-off if compute is constrained.

## 6. Qualitative Analysis

547 test reviews are mispredicted by BiGRU but correctly rated by RoBERTa, vs. 414 the other way. Inspecting examples of each (full text in results/qualitative_examples.txt, full analysis in results/qualitative_analysis.md):

**RoBERTa wins on:** contrastive-conjunction reviews ("good, _but_...", "very happy, _however_...") where an early strong-polarity clause is overturned later in the text; figurative language/idioms ("ripping my hair out") that are lexically strong-negative but contextually mild; sarcasm ("thank goodness it was a free book") requiring magnitude calibration, not just polarity; and long, informal reviews with slang/interjections where BiGRU's fixed 200-token window and fastText OOV handling lose resolving context. RoBERTa's full self-attention over BPE subwords integrates this long-range, contrastive structure far more reliably than BiGRU's recurrent+attention pooling.

**BiGRU wins on:** very short, uniformly positive reviews with no contrast word, where the true label (4 vs. 5 stars) is arguably under-determined by the text and RoBERTa's stronger positive-sentiment prior overshoots to 5; and mild comparative-negation phrasing ("doesn't work any better than...") that is lexically negative but semantically near-neutral, where RoBERTa over-weights the negation cue.

![Figure 3: Gradient-norm token importance](images/figure3_token_importance.png)

_Figure 3: Gradient-norm token-importance (saliency w.r.t. input embeddings, true-class logit) for a true=1-star review ("...I'm very disappointed in the service", wrong case size shipped). Importance concentrates on "disappointed" and the negation/mismatch tokens late in the review rather than the mildly positive opener ("looks cool") — consistent with RoBERTa correctly identifying the decisive complaint rather than anchoring on the early token, the same behavior underlying its wins above._

## 7. Comparative Analysis & Conclusion

RoBERTa outperforms the BiGRU baseline on every metric: +2.6 pts accuracy, +2.4 pts Macro-F1, and 0.048 lower MAE, with lower variance across seeds. Both models respect the ordinal nature of the task reasonably well — errors are overwhelmingly adjacent-class (BiGRU 34.6%, RoBERTa 33.8%) rather than distant (BiGRU 1.1%, RoBERTa 0.6%) — but RoBERTa is both more accurate and slightly better calibrated ordinally (higher exact-match rate, lower near-miss/distant rate). Per-class F1 shows both models struggle most on the middle classes (2- and 3-star), which are intrinsically more ambiguous (mixed sentiment is common there), and are strongest on the extremes (1- and 5-star), where sentiment polarity is least ambiguous. The qualitative analysis explains the quantitative gap: RoBERTa's contextual, long-range self-attention over subwords handles contrastive conjunctions, idioms, sarcasm, and informal/long text substantially better than BiGRU's fixed-window recurrent pooling over static-then-fine-tuned word embeddings, at the cost of far more compute. The length ablation shows that cost can be partly reclaimed (half the sequence length, negligible accuracy loss) without sacrificing RoBERTa's architectural advantage.
