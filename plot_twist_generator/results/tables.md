## Dataset

```json
{
  "seed": 42,
  "n_train": 20000,
  "n_val": 1000,
  "n_test": 1000,
  "vocab_size": 7856,
  "min_freq": 3,
  "max_vocab": 10000,
  "max_story_tokens": 300,
  "train_tokens": 3862623,
  "train_unk_rate": 0.0012841015030459872,
  "mean_story_len": 193.13115,
  "tokenisation": "word-level, lower-case"
}
```

## Model comparison (Part 2 / 5)

| Model | Params | Epochs run | Train time (min) | Best val loss | Val PPL | Test PPL |
|---|---|---|---|---|---|---|
| Vanilla RNN | 6,435,504 | 20 | 16.9 | 2.899 | 18.15 | 18.62 |
| LSTM | 7,618,224 | 20 | 22.4 | 2.487 | 12.03 | 12.25 |

## Generation metrics on held-out test prompts (Part 5)

Settings: `{"n_prompts": 200, "prompt_len": 20, "max_new": 200, "seed": 2024}`. Rep-3 = fraction of repeated trigram occurrences within a story; premature = EOS before 20 generated tokens; overlong = no EOS within budget.

| Model | Decoding | Distinct-1 | Distinct-2 | Rep-3 | Avg len | Premature | Overlong | UNK rate |
|---|---|---|---|---|---|---|---|---|
| Vanilla RNN | greedy | 0.009 | 0.023 | 0.488 | 143.1 | 1.00% | 57.00% | 0.00% |
| Vanilla RNN | temp0.5 | 0.035 | 0.144 | 0.090 | 127.8 | 0.50% | 34.50% | 0.01% |
| Vanilla RNN | temp0.8 | 0.057 | 0.286 | 0.022 | 136.7 | 0.50% | 35.00% | 0.06% |
| Vanilla RNN | temp1.2 | 0.099 | 0.554 | 0.003 | 132.2 | 2.50% | 28.50% | 0.17% |
| Vanilla RNN | topk10 | 0.039 | 0.217 | 0.024 | 131.4 | 1.00% | 34.00% | 0.00% |
| Vanilla RNN | topk40 | 0.055 | 0.315 | 0.013 | 129.7 | 1.00% | 31.50% | 0.05% |
| LSTM | greedy | 0.021 | 0.081 | 0.359 | 153.0 | 0.00% | 34.00% | 0.00% |
| LSTM | temp0.5 | 0.037 | 0.168 | 0.118 | 156.7 | 0.00% | 31.00% | 0.02% |
| LSTM | temp0.8 | 0.055 | 0.276 | 0.041 | 160.6 | 0.00% | 34.50% | 0.06% |
| LSTM | temp1.2 | 0.089 | 0.507 | 0.006 | 163.1 | 0.00% | 38.00% | 0.24% |
| LSTM | topk10 | 0.045 | 0.246 | 0.040 | 166.5 | 0.00% | 38.00% | 0.01% |
| LSTM | topk40 | 0.056 | 0.315 | 0.022 | 166.9 | 1.00% | 41.50% | 0.09% |

## Blinded human evaluation (raters: rater1, rater2, rater3)

| System | Coherence | Grammar | Relevance | Creativity |
|---|---|---|---|---|
| Vanilla RNN | 1.40 ± 0.49 | 2.47 ± 0.62 | 1.10 ± 0.30 | 1.00 ± 0.00 |
| LSTM | 1.50 ± 0.50 | 2.50 ± 0.56 | 1.20 ± 0.40 | 1.17 ± 0.37 |

Inter-rater agreement:

| Criterion | Mean abs diff | Exact | Within ±1 |
|---|---|---|---|
| coherence | 0.37 | 63% | 100% |
| grammar | 0.37 | 63% | 100% |
| relevance | 0.30 | 70% | 100% |
| creativity | 0.10 | 90% | 100% |