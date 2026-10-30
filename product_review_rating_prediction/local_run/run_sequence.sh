#!/bin/bash
set -e
cd "$(dirname "$0")"
python 02_train_bigru.py --seed 42
python 02_train_bigru.py --seed 123
python 03_train_roberta.py --variant main       --seed 42
python 03_train_roberta.py --variant main       --seed 123
python 03_train_roberta.py --variant ablation_b --seed 42
python 04_analyze.py
