#!/usr/bin/env bash
# Full reproduction (seeds fixed inside each script). ~1-2 h on a single modest GPU.
set -e
pip install -r requirements.txt
python data_prep.py                       # Part 1  (use --toy for a 1-minute offline smoke test)
python train.py --cell rnn  --seed 42     # Part 2
python train.py --cell lstm --seed 42
python generate.py                        # Parts 3 & 4
python evaluate_generation.py             # Part 5 quantitative
python make_human_eval.py                 # Part 5 human: creates blinded files in human_eval/
# --> give human_eval/ratings_template.csv + INSTRUCTIONS.md to >=3 raters, save as ratings_rater1..3.csv
# optional: python llm_extension.py && python make_human_eval.py --include_llm
python aggregate_human_eval.py
python make_report_tables.py              # -> results/tables.md, results/loss_curves.png
