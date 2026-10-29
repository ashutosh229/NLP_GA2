# Q3 Report — Abstractive Text Summarization

## Objective

This project addresses Question 3 from the NLP Group Assignment: building a full Transformer encoder-decoder model for abstractive summarization on the CNN/DailyMail dataset. The task requires a model implemented from scratch, with the exception of a pretrained tokenizer, and evaluation using standard summarization metrics.

## Dataset and preprocessing

The notebook uses the Hugging Face CNN/DailyMail dataset version 3.0.0, keeping the published train/validation/test splits separate:

- Training: 287,113 examples
- Validation: 13,368 examples
- Test: 11,490 examples

The reference summaries are stored as multi-sentence highlights with newline separators, which is required for ROUGE-Lsum evaluation. The preprocessing pipeline uses a shared `bert-base-uncased` WordPiece tokenizer, lowercasing is preserved through the uncased vocabulary, and the article/summary lengths are truncated to 512 and 128 tokens respectively.

## Model architecture

The model is a standard Transformer sequence-to-sequence architecture implemented from scratch with basic tensor operations and `nn.Linear`, `nn.Embedding`, and `nn.LayerNorm`. It includes:

- sinusoidal positional encodings
- scaled dot-product attention with masking
- multi-head self-attention in the encoder and decoder
- encoder-decoder attention in the decoder
- feed-forward layers and residual connections
- padding and causal masks
- final vocabulary projection layer

The final architecture uses:

- 2 encoder layers
- 2 decoder layers
- `d_model = 128`
- 4 attention heads
- `d_ff = 512`
- dropout = 0.1

This gives a model with 8,769,850 trainable parameters.

## Training setup

The model was trained on the full training split using teacher forcing and a padding-ignored cross-entropy objective. The optimization setup follows the paper's Transformer schedule:

- Optimizer: Adam with `beta1 = 0.9`, `beta2 = 0.98`, `eps = 1e-9`
- LR schedule: warmup = 4,000, inverse-square-root decay with scale 0.5
- Effective batch size: 16
- Mixed precision: enabled on CUDA
- Early stopping: based on validation loss
- Best validation checkpoint: epoch 8
- Best validation loss: 5.1243

The notebook also saves the best model, optimizer state, tokenizer files, loss curves, per-example predictions, evaluation JSON, and attention visualization.

## Quantitative evaluation

The final run was evaluated on the full official test split. The assignment reference Lead-3 baseline and the run's own Lead-3 baseline are reported below alongside the Transformer results.

| System | Test examples | ROUGE-1 | ROUGE-2 | ROUGE-L | ROUGE-Lsum |
|---|---:|---:|---:|---:|---:|
| Lead-3 (assignment reference) | 11,490 | 40.06 | 17.48 | 25.02 | 36.36 |
| Lead-3 (this run) | 11,490 | 39.91 | 17.38 | 24.90 | 36.20 |
| Scratch Transformer | 11,490 | 22.84 | 4.37 | 15.94 | 21.38 |

These figures are F1 scores. The model clearly stays below the Lead-3 baseline, which is expected for a compact from-scratch Transformer without pretrained summarization weights. The run still satisfies the assignment requirements by producing a complete encoder-decoder pipeline and reporting quantitative results on the full test split.

## Qualitative analysis

The generated summaries are readable but often generic and weakly grounded. The most frequent issues are vague introductions, hallucinated names, omitted essential facts, and weak handling of numbers and institutions.

### Example 1: medical marijuana article

- Article topic: legalisation and medical-marijuana advocacy
- Generated summary: short, generic, and incomplete; it repeated phrases instead of identifying the central argument
- Observation: the model captured the topic but failed to include the main actionable claim or key names

### Example 2: child with Twitter fame and gang-life imagery

- Generated output: vague references to a boy on Twitter and a video
- Observation: the summary missed the central event and the notable social-media / backlash details

### Example 3: Chris Christie town-hall controversy

- Generated output: highly repetitive and semantically unstable
- Observation: the model did not preserve the event framing, the teacher's role, or the key controversy

### Example 4: Cassey Ho body-image video

- Generated output: very general and not grounded in the article's specific argument
- Observation: the model omitted the main point that she was responding to cruel criticism about her body image

### Example 5: general summary drift

- In several examples, the model began with generic filler such as "the first time" or an article-level phrase rather than naming the key subject
- Observation: this is a common failure mode for smaller from-scratch seq2seq models and explains the low ROUGE-2 score

## Interpretation

The Transformer learns some surface-level summarization behaviour, but it does not yet reliably select the correct factual content from long articles. The weak performance is particularly visible in entity-heavy and argument-heavy news stories, where the model tends to over-generate generic phrasing and under-specify the main message.

The attention plots and prediction outputs indicate that the model is modelling structure and token-level dependencies, but it has not yet matched the factual precision of a Lead-3 baseline or a pretrained summarization system. The assignment requirement is therefore met technically, but the final quality remains limited by model scale and training setup.

## Final conclusion

The project successfully implements a from-scratch Transformer summarization architecture, trains it on the required CNN/DailyMail split, and reports the requested ROUGE metrics and qualitative examples. The resulting model is functional but clearly underperforms the Lead-3 baseline, which is a realistic outcome for a compact, non-pretrained encoder-decoder trained on a challenging summarization task.
