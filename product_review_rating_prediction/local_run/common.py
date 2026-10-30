"""Shared paths/constants for the local (desktop GPU) training pipeline.

Big downloads/caches are kept off the near-full root disk, on /data3 instead.
Import this before any `transformers`/`datasets` import so HF_HOME takes effect.
"""
import os

CACHE_ROOT = '/data3/Aditya/nlpga2_cache'
os.environ.setdefault('HF_HOME', f'{CACHE_ROOT}/hf')

DATA_DIR    = f'{CACHE_ROOT}/data'     # vocab, embedding matrix, encoded arrays, tokenized arrow datasets
RESULTS_DIR = f'{CACHE_ROOT}/results'  # checkpoints, metrics json, figures
FASTTEXT_DIR = f'{CACHE_ROOT}/fasttext'

for d in (DATA_DIR, RESULTS_DIR, FASTTEXT_DIR):
    os.makedirs(d, exist_ok=True)

NUM_CLASSES = 5
SEEDS       = [42, 123]

DATASET_NAME = 'SetFit/amazon_reviews_multi_en'
ROBERTA_MODEL = 'roberta-base'

MAX_SEQ_LEN_CLASSIC = 200
MIN_FREQ    = 3
EMBED_DIM   = 300
PAD_TOKEN   = '<PAD>'
UNK_TOKEN   = '<UNK>'

FASTTEXT_VEC_URL = 'https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip'
FASTTEXT_LOCAL   = f'{FASTTEXT_DIR}/crawl-300d-2M.vec'
