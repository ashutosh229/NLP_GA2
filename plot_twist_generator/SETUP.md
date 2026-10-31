# SETUP: Plot-Twist Story Generator

This guide explains how to install, run and reproduce everything in `plot_twist_generator/`.

All commands below are run from inside the `plot_twist_generator/` directory, because the scripts import
each other with plain `from common import *` and write to folders next to themselves.

---

## 1. Requirements

| Requirement | Notes                                                                                                    |
| ----------- | -------------------------------------------------------------------------------------------------------- |
| Python      | 3.10 or newer recommended                                                                                |
| OS          | Linux, macOS or Windows (Windows users: see the encoding note in Troubleshooting)                        |
| GPU         | Optional but strongly recommended. A single modest CUDA GPU trains both models in about 40 minutes total |
| Disk        | About 1 GB for the TinyStories download and Hugging Face cache; checkpoints are about 30 MB each         |
| Internet    | Needed once, to download TinyStories (and SmolLM2 if you run the optional LLM step)                      |

Python packages (`requirements.txt`): `torch`, `torchvision`, `datasets>=2.14`, `matplotlib`, `numpy`,
and `transformers>=4.40` (only for the optional LLM extension).

## 2. Installation

```bash
git clone https://github.com/ashutosh229/NLP_GA2.git
cd NLP_GA2/plot_twist_generator

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
```

**Install PyTorch.** Pick the line that matches your machine, then install the rest:

```bash
# NVIDIA GPU (CUDA 12.6 wheels, as used in run_all.sh)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# CPU only
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# then everything else
pip install -r requirements.txt
```

Verify the install:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## 3. Quick smoke test (about 1 to 2 minutes, no download)

The `--toy` flag replaces TinyStories with a tiny synthetic corpus so you can check that the whole pipeline works.

```bash
python data_prep.py --toy
python train.py --cell rnn  --epochs 2
python train.py --cell lstm --epochs 2
python generate.py --n_samples 1 --max_new 50
```

> **Warning:** the toy run writes into the same `processed/`, `checkpoints/` and `results/` folders as the real run.
> After the smoke test, re-run the full pipeline from step 4.1 to overwrite the toy artefacts.

## 4. Full reproduction

`run_all.sh` runs everything end to end (it also installs the CUDA build of PyTorch, so skip it if you installed manually).
The individual steps are:

### 4.1 Part 1: Data preparation

```bash
python data_prep.py
```

Downloads `roneneldan/TinyStories`, samples 20,000 train and 2,000 held-out stories, cleans, de-duplicates, tokenises,
builds the vocabulary, and writes to `processed/`:

| File                                  | Content                                                       |
| ------------------------------------- | ------------------------------------------------------------- |
| `vocab.json`                          | 7,856 tokens (4 special + words with frequency of at least 3) |
| `train.json`, `val.json`, `test.json` | stories as lists of token ids (`<bos>` ... `<eos>`)           |
| `test_texts.json`                     | raw text of test stories                                      |
| `meta.json`                           | dataset statistics (sizes, UNK rate, mean length, seed)       |

Useful options: `--n_train`, `--n_val`, `--n_test`, `--min_freq`, `--max_vocab`, `--min_story_tokens`,
`--max_story_tokens`, `--seed`, `--toy`.

### 4.2 Part 2: Train the models

```bash
python train.py --cell rnn  --seed 42
python train.py --cell lstm --seed 42
```

Each run saves the best-validation checkpoint to `checkpoints/<cell>.pt` and the full training history
(per-epoch loss and perplexity, parameters, timing, test perplexity) to `results/history_<cell>.json`.

Reference timings (single GPU): RNN about 17 min, LSTM about 22 min for 20 epochs.

Options: `--emb 256 --hid 512 --layers 1 --dropout 0.3 --lr 2e-3 --bs 64 --seq_len 128 --epochs 20 --patience 3 --clip 1.0`.

### 4.3 Parts 3 & 4: Generate stories

```bash
python generate.py                     # defaults: --n_samples 3 --max_new 200 --seed 1234
```

Produces, in `results/`:

| File                                                   | Content                                                                       |
| ------------------------------------------------------ | ----------------------------------------------------------------------------- |
| `generations_part3.json`, `generated_stories_part3.md` | 5 theme prompts × 2 models × 6 decoding configs (decoding-strategy study)     |
| `generations_part4.json`, `generated_stories_part4.md` | 5 plot-twist prompts × 2 models × 2 configs (`temp0.8`, `topk40`) × 2 samples |

The Markdown files contain a `**Commentary:**` placeholder after every sample, where you write your own analysis.
Anything inside a `MODEL OUTPUT` block is raw model text and should not be edited.

Prompts and decoding configs are defined in `prompts.py`.

### 4.4 Part 5: Quantitative evaluation

```bash
python evaluate_generation.py          # defaults: 200 test prompts of 20 tokens, --max_new 200, --seed 2024
```

Writes `results/gen_metrics.json` with Distinct-1/2, Rep-3, average length, premature-EOS rate, overlong rate and UNK rate
for every model × decoding config.

### 4.5 Part 5: Blinded human evaluation

```bash
python make_human_eval.py
```

This shuffles the Part 4 samples (sample index 0 only), hides which model or config wrote each one, and creates in `human_eval/`:

- `ratings_template.csv`: the blinded sheet (`S01`, `S02`, ...)
- `key.json`: the secret mapping from sample id to model/config/prompt (**do not give this to raters**)
- `INSTRUCTIONS.md`: the rating rubric

Workflow:

1. Give each rater **only** `ratings_template.csv` and `INSTRUCTIONS.md` (at least 3 raters recommended).
2. Each rater copies the template to `ratings_raterN.csv` (`ratings_rater1.csv`, `ratings_rater2.csv`, ...), fills the
   four integer score columns (1 to 5) for **coherence, grammar, relevance, creativity**, and does not edit other columns.
3. Put all rating files in `human_eval/`, then aggregate:

```bash
python aggregate_human_eval.py
```

Writes `results/human_eval_summary.json` (mean and std per model and per model/config, per-rater means, and inter-rater agreement:
mean absolute difference, exact agreement, within ±1).

> **Warning:** do **not** re-run `make_human_eval.py` after raters have started. It re-shuffles and overwrites `key.json`
> and `ratings_template.csv`, which would break the link between existing rating files and the models.
> Also, if you retrain the models or change generation settings, the existing ratings no longer match the new samples.

### 4.6 Report tables and figures

```bash
python make_report_tables.py
```

Writes `results/tables.md` (dataset info, model comparison, generation metrics, human-evaluation tables) and
`results/loss_curves.png` (train/val loss and validation perplexity per epoch).

### 4.7 Optional: LLM baseline

```bash
python llm_extension.py                          # default: HuggingFaceTB/SmolLM2-360M-Instruct
python llm_extension.py --ckpt <other-hf-model>  # any chat-template causal LM
python make_human_eval.py --include_llm          # blind the LLM samples into a NEW rating round
```

Generates 2 samples per prompt for `temp0.8` and `topk40` using a fixed prompt template (zero-shot, not fine-tuned),
and saves them with metadata (checkpoint, template, decoding, precision, hardware) to `results/llm_part4.json` and
`results/llm_metadata.json`. Remember the warning in section 4.5: adding the LLM samples means re-blinding and re-rating.

## 5. Output map

| Path                                           | Created by                | Description                                        |
| ---------------------------------------------- | ------------------------- | -------------------------------------------------- |
| `processed/`                                   | `data_prep.py`            | vocabulary, tokenised splits, dataset metadata     |
| `checkpoints/{rnn,lstm}.pt`                    | `train.py`                | best-validation weights plus model config          |
| `results/history_{rnn,lstm}.json`              | `train.py`                | training curves, timing, val/test perplexity       |
| `results/generations_part{3,4}.json`           | `generate.py`             | all samples with prompt, seed, tokens and EOS flag |
| `results/generated_stories_part{3,4}.md`       | `generate.py`             | readable versions with commentary placeholders     |
| `results/gen_metrics.json`                     | `evaluate_generation.py`  | quantitative generation metrics                    |
| `results/llm_part4.json`, `llm_metadata.json`  | `llm_extension.py`        | LLM baseline samples and settings                  |
| `human_eval/`                                  | `make_human_eval.py`      | blinded sheet, key, instructions, rater CSVs       |
| `results/human_eval_summary.json`              | `aggregate_human_eval.py` | human-evaluation statistics                        |
| `results/tables.md`, `results/loss_curves.png` | `make_report_tables.py`   | report-ready tables and figure                     |

## 6. Reproducibility

- Seeds: data sampling `--seed 42`, training `--seed 42`, `generate.py` `--seed 1234`, `evaluate_generation.py` `--seed 2024`,
  human-eval shuffle `--seed 7`.
- Every generated sample stores its own seed, so any individual output can be regenerated.
- GPU results can differ slightly across hardware, CUDA and PyTorch versions even with fixed seeds.
- Greedy decoding is run once per prompt; sampled configs use several seeds (`seed + sample_index`).

## 7. Troubleshooting

| Problem                                                                                     | Fix                                                                                                                                                                                                                                    |
| ------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ModuleNotFoundError: common` or files written in the wrong place                           | Run scripts from inside `plot_twist_generator/`                                                                                                                                                                                        |
| `FileNotFoundError: processed/vocab.json` or `train.json`                                   | Run `data_prep.py` first                                                                                                                                                                                                               |
| `FileNotFoundError: checkpoints/rnn.pt`                                                     | Run `train.py --cell rnn` and `--cell lstm` before `generate.py` / `evaluate_generation.py`                                                                                                                                            |
| `no human_eval/ratings_rater*.csv files found`                                              | Create rating files as described in section 4.5 before running `aggregate_human_eval.py`                                                                                                                                               |
| CUDA out of memory                                                                          | Lower `--bs` (for example 32) or `--hid`; or train on CPU (`CUDA_VISIBLE_DEVICES=""`)                                                                                                                                                  |
| Training is very slow                                                                       | You are on CPU. Use a GPU, or reduce `--n_train` in `data_prep.py` and `--epochs` in `train.py`                                                                                                                                        |
| Dataset download fails or is slow                                                           | Check your connection; `datasets` caches in `~/.cache/huggingface`. Use `--toy` to test offline                                                                                                                                        |
| `UnicodeDecodeError: 'utf-8' codec can't decode byte 0xb1` when reading `results/tables.md` | On Windows, `make_report_tables.py` writes the "±" symbol with the system code page. Change its last write to `(RES / "tables.md").write_text("\n".join(md), encoding="utf-8")` and re-run. On Windows you can also set `PYTHONUTF8=1` |
| Windows: garbled quotes in rater CSVs                                                       | Open and save CSVs as UTF-8 (the scripts read them as UTF-8)                                                                                                                                                                           |
| `torch.load` warnings about `weights_only`                                                  | Safe to ignore here; checkpoints contain only a state dict and a plain config dict                                                                                                                                                     |
| LLM step cannot download the model                                                          | Check network access to Hugging Face, or pass a locally downloaded model path via `--ckpt`                                                                                                                                             |

## 8. Notes on interpreting outputs

- Text is lower-cased and word-tokenised, so generated stories are re-capitalised heuristically by `detokenize()`. Capitalisation
  and spacing around quotes are approximate and are not model errors.
- Generations that reach the token budget without `<eos>` are flagged `ended: false` and counted as "overlong" in the metrics.
- Rare words map to `<unk>`; it is excluded from `<pad>`/`<bos>` blocking but not from sampling, so `<unk>` can appear in outputs
  (more often at high temperature).
