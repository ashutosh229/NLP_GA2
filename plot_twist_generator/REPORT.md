# Plot-Twist Story Generator (NLP_GA2)

A from-scratch study of **word-level recurrent language models** (Vanilla RNN vs. LSTM) trained on the
[TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) corpus and used to continue short story
openings, with a special focus on the **"plot-twist" challenge**: can a small recurrent model take a human-written
opening that sets up a surprising situation and continue it sensibly?

The project covers the full pipeline: data preparation, model training, decoding-strategy comparison, quantitative
generation metrics, a blinded human evaluation, and an optional pre-trained LLM baseline.

> **Setup and run instructions:** see [`plot_twist_generator/SETUP.md`](plot_twist_generator/SETUP.md).
> **Written report:** `plot_twist_generator/REPORT.md`.

---

## 1. What the system does

1. **Prepares data.** Samples 20,000 train / 1,000 validation / 1,000 test stories from TinyStories, lower-cases and
   tokenises them at word level, and builds a 7,856-token vocabulary.
2. **Trains two language models** with identical hyper-parameters: a tanh Vanilla RNN and an LSTM.
3. **Generates story continuations** from prompts using six decoding configurations (greedy, three temperatures, two top-k settings).
4. **Runs a plot-twist challenge** on five hand-written prompts (exam, robot, detective, village, letter).
5. **Evaluates** generations quantitatively (diversity, repetition, length and termination behaviour, UNK rate) and
   with a **blinded 3-rater human evaluation** (coherence, grammar, relevance, creativity).
6. *(Optional)* Generates the same plot-twist continuations with a small instruction-tuned LLM
   (`HuggingFaceTB/SmolLM2-360M-Instruct`) as a reference point, and blinds them into the human evaluation.

## 2. Pipeline overview

```
data_prep.py ──► processed/ (vocab, train/val/test ids)
      │
train.py --cell rnn|lstm ──► checkpoints/{rnn,lstm}.pt + results/history_*.json
      │
generate.py ──────────────► results/generations_part{3,4}.json + generated_stories_part{3,4}.md
evaluate_generation.py ───► results/gen_metrics.json
make_human_eval.py ───────► human_eval/ (blinded template, key.json, INSTRUCTIONS.md)
      │   (raters fill ratings_rater1..N.csv)
aggregate_human_eval.py ──► results/human_eval_summary.json
make_report_tables.py ────► results/tables.md + results/loss_curves.png
llm_extension.py (opt.) ──► results/llm_part4.json + llm_metadata.json
```

## 3. Model and data details

| Item | Value |
|---|---|
| Tokenisation | word-level, lower-cased; regex keeps words, contractions and basic punctuation |
| Special tokens | `<pad>`, `<unk>`, `<bos>`, `<eos>` |
| Vocabulary | 7,856 tokens (min frequency 3, cap 10,000) |
| Stories | 20,000 train / 1,000 val / 1,000 test (de-duplicated; test and val never overlap train) |
| Story length | 30 to 300 tokens (longer stories truncated, no `<eos>` added when truncated); mean 193.1 |
| Train tokens / UNK rate | 3,862,623 / 0.128% |
| Architecture | Embedding (256) → 1-layer RNN or LSTM (hidden 512) → Dropout (0.3) → Linear to vocab |
| Parameters | RNN 6,435,504; LSTM 7,618,224 |
| Optimiser | Adam, lr 2e-3, `ReduceLROnPlateau` (factor 0.5, patience 1), grad-clip 1.0 |
| Training | batch 64, 128-token chunks, up to 20 epochs, early stopping (patience 3), seed 42 |

**Decoding configurations** (`prompts.py`): `greedy`, `temp0.5`, `temp0.8`, `temp1.2`, `topk10`, `topk40`.
Generation is seeded, so every output is reproducible from its recorded seed. `<pad>` and `<bos>` are never sampled.

## 4. Results

### Language-model quality

| Model | Params | Epochs | Train time | Val PPL | Test PPL |
|---|---|---|---|---|---|
| Vanilla RNN | 6.44 M | 20 | 16.9 min | 18.15 | 18.62 |
| LSTM | 7.62 M | 20 | 22.4 min | 12.03 | 12.25 |

The LSTM reaches about **34% lower test perplexity** than the RNN. Both models used all 20 epochs; the LSTM's
validation loss was still falling slowly, so more epochs would likely help it further.

### Generation metrics (200 held-out test prompts of 20 tokens, up to 200 new tokens)

Distinct-n is computed over the pooled set of generations. Rep-3 is the mean fraction of repeated trigrams per story.

| Model | Decoding | Distinct-1 | Distinct-2 | Rep-3 | Avg len | Overlong | UNK rate |
|---|---|---|---|---|---|---|---|
| RNN | greedy | 0.009 | 0.023 | 0.488 | 143.1 | 57.0% | 0.00% |
| RNN | temp 0.8 | 0.057 | 0.286 | 0.022 | 136.7 | 35.0% | 0.06% |
| RNN | temp 1.2 | 0.099 | 0.554 | 0.003 | 132.2 | 28.5% | 0.17% |
| RNN | top-k 40 | 0.055 | 0.315 | 0.013 | 129.7 | 31.5% | 0.05% |
| LSTM | greedy | 0.021 | 0.081 | 0.359 | 153.0 | 34.0% | 0.00% |
| LSTM | temp 0.8 | 0.055 | 0.276 | 0.041 | 160.6 | 34.5% | 0.06% |
| LSTM | temp 1.2 | 0.089 | 0.507 | 0.006 | 163.1 | 38.0% | 0.24% |
| LSTM | top-k 40 | 0.056 | 0.315 | 0.022 | 166.9 | 41.5% | 0.09% |

All six configurations per model are in `plot_twist_generator/results/gen_metrics.json`.
"Overlong" means no `<eos>` was produced within the 200-token budget.

### Blinded human evaluation (3 raters, 20 samples, 1 to 5 scale)

| System | Coherence | Grammar | Relevance | Creativity |
|---|---|---|---|---|
| Vanilla RNN | 1.40 ± 0.49 | 2.47 ± 0.62 | 1.10 ± 0.30 | 1.00 ± 0.00 |
| LSTM | 1.50 ± 0.50 | 2.50 ± 0.56 | 1.20 ± 0.40 | 1.17 ± 0.37 |

Inter-rater agreement is high within ±1 point (100% on every criterion); exact agreement ranges from 63% (coherence, grammar)
to 90% (creativity).

## 5. Key findings

- **LSTM beats the RNN on perplexity** by a clear margin, and also produces slightly longer, more fluent text.
- **Greedy decoding degenerates** into repetition (Rep-3 of 0.49 for the RNN, 0.36 for the LSTM) and has the lowest diversity.
- **Higher temperature raises diversity** (Distinct-2 rises from about 0.14 at T=0.5 to about 0.55 at T=1.2) but also raises the UNK rate and reduces coherence.
  Top-k 40 gives diversity similar to temperature 0.8 with a lower repetition rate than pure sampling at the same diversity.
- **Plot twists are not achieved by either model.** Human ratings are low across the board (coherence about 1.4 to 1.5 / 5,
  relevance about 1.1 to 1.2 / 5, creativity about 1.0 to 1.2 / 5). The continuations often drift to unrelated characters (for example "Lily" or "Tom")
  and generic TinyStories endings, ignoring the opening's twist. Differences between the two models in the human study
  are small relative to the spread, with only 20 samples.
- **Many generations never terminate.** 28% to 57% of outputs hit the 200-token budget without emitting `<eos>`.

## 6. Repository layout

```
.
├── README.md
├── LICENSE
└── plot_twist_generator/
    ├── SETUP.md                  # install / run / troubleshooting
    ├── REPORT.md                 # written report
    ├── requirements.txt
    ├── run_all.sh                # full reproduction script
    ├── common.py                 # paths, seeding, tokeniser, Vocab
    ├── data_prep.py              # Part 1: dataset and vocabulary
    ├── models.py                 # RecurrentLM (RNN / LSTM)
    ├── train.py                  # Part 2: training and evaluation
    ├── prompts.py                # prompts and decoding configs
    ├── generate.py               # Parts 3 & 4: decoding + plot-twist generation
    ├── metrics.py                # distinct-n, repetition, termination stats
    ├── evaluate_generation.py    # Part 5: quantitative evaluation
    ├── make_human_eval.py        # Part 5: blinded rating sheets
    ├── aggregate_human_eval.py   # Part 5: rater statistics and agreement
    ├── make_report_tables.py     # tables.md and loss_curves.png
    ├── llm_extension.py          # optional SmolLM2 baseline
    ├── processed/                # vocab + tokenised splits (generated)
    ├── checkpoints/              # model weights (generated)
    ├── results/                  # histories, generations, metrics, tables (generated)
    └── human_eval/               # blinded sheets, key, ratings
```

## 7. Quick start

```bash
git clone https://github.com/ashutosh229/NLP_GA2.git
cd NLP_GA2/plot_twist_generator
pip install -r requirements.txt
python data_prep.py && python train.py --cell rnn && python train.py --cell lstm && python generate.py
```

Full, step-by-step instructions (GPU/CPU installs, smoke test, human-evaluation workflow, troubleshooting) are in
[`plot_twist_generator/SETUP.md`](plot_twist_generator/SETUP.md).

## 8. Reproducibility

Seeds are fixed in every script (data sampling 42, training 42, generation 1234, evaluation 2024, human-eval shuffle 7).
Each generated sample records its seed. Exact numbers can still vary slightly across hardware and PyTorch versions
(especially on GPU).

## 9. License

Released under the MIT License. See [`LICENSE`](LICENSE). Copyright (c) 2026 Ashutosh Kumar Jha.
