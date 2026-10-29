# Setup and Execution Instructions

## Question 1: Attention-Based Neural Machine Translation (English to Hindi)

This repository contains the complete implementation for building, training, evaluating, and visualizing an **English-to-Hindi Neural Machine Translation (NMT)** system. It compares a standard **LSTM Encoder–Decoder Baseline (without Attention)** against an **LSTM Encoder–Decoder with Bahdanau Additive Attention**.

---

### 1. System Requirements & Environment
- **Operating System:** Windows / Linux / macOS
- **Python Version:** Python 3.10 or 3.11 (Python 3.11.4 recommended)
- **Hardware:** Multi-core CPU or NVIDIA CUDA-enabled GPU

---

### 2. Environment Setup

#### Step 2.1: Create and Activate Virtual Environment
Open PowerShell or your preferred terminal in this directory (`neural_machine_translation`):

```powershell
# Using Python 3.11
python -m venv venv

# Activate Virtual Environment
# Windows PowerShell:
.\venv\Scripts\Activate.ps1

# Windows Command Prompt:
.\venv\Scripts\activate.bat

# Linux / macOS:
source venv/bin/activate
```

#### Step 2.2: Install Required Dependencies
Install all required packages from `requirements.txt`:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

The key dependencies include:
- `torch>=2.0.0` (PyTorch Seq2Seq deep learning framework)
- `datasets>=2.14.0` (Hugging Face Datasets for official IIT Bombay parallel corpus)
- `sacrebleu>=2.3.0` (Standardized machine translation BLEU scoring)
- `nltk>=3.8.0` (NLP tokenization utilities)
- `matplotlib>=3.7.0` & `seaborn>=0.12.0` (Attention heatmaps and training curves)
- `pandas>=2.0.0`, `numpy>=1.24.0`, `scikit-learn>=1.2.0`, `tqdm>=4.65.0`

---

### 3. Step-by-Step Execution Guide

#### Step 3.1: Data Preparation & Preprocessing
Loads the official **IIT Bombay English–Hindi Parallel Corpus** (`cfilt/iitb-english-hindi`), normalizes English and Hindi texts, separates Devanagari punctuation (such as `।`), tokenizes sentences, adds `<SOS>` and `<EOS>` tokens, filters sentence lengths, builds source and target vocabularies, and caches the dataset:

```bash
python data_preprocessing.py
```
*Outputs generated:*
- `data/preprocessed_25000_30.pkl`: Serialized preprocessed dataset splits and vocabularies.
- `data/corpus_stats.json`: Corpus statistics (vocabulary sizes, sequence length bounds, sample counts).

#### Step 3.2: Model Training
Trains both **Model A (Baseline LSTM without Attention)** and **Model B (Attention-based LSTM with Bahdanau Attention)** under identical conditions (same seed, dimensions, optimizer, teacher-forcing ratio, and epochs):

```bash
python train.py --epochs 6 --batch_size 64 --embed_dim 128 --hidden_dim 256
```
*Outputs generated:*
- `checkpoints/baseline_lstm_best.pt`: Best checkpoint for Model A (minimum validation loss).
- `checkpoints/attention_lstm_best.pt`: Best checkpoint for Model B (minimum validation loss).
- `checkpoints/training_history.json`: Epoch-by-epoch loss and token accuracy progression.

#### Step 3.3: Evaluation & Length-Stratified Analysis
Evaluates both models on the held-out test set using greedy autoregressive decoding. Computes overall corpus BLEU and evaluates performance across sentence length categories (`1–5`, `6–10`, `11–15`, `16+` tokens):

```bash
python evaluate.py
```
*Outputs generated:*
- `data/evaluation_results.json`: Quantitative BLEU scores, length bucket breakdowns, and relative improvements.

#### Step 3.4: Attention Visualizations & Error Analysis
Generates training curves, sentence length analysis graphs, and attention heatmaps (for successful translations and 5 classified error cases):

```bash
python visualize_attention.py
```
*Outputs generated in `plots/`:*
- `plots/training_curves.png`: Training & validation loss and accuracy comparison.
- `plots/sentence_length_analysis.png`: BLEU score vs. sentence length bar chart.
- `plots/attention_sample_1.png` - `attention_sample_3.png`: Attention heatmaps for successful translations.
- `plots/error_case_1_missing_word.png` to `plots/error_case_5_*.png`: Classified error attention heatmaps.
- `data/error_analysis.json`: Detailed error analysis metadata.

---

### 4. File Structure
```
neural_machine_translation/
├── data/                                # Preprocessed dataset caches and metrics
│   ├── corpus_stats.json
│   ├── evaluation_results.json
│   └── error_analysis.json
├── checkpoints/                         # Model weights and training history
│   ├── baseline_lstm_best.pt
│   ├── attention_lstm_best.pt
│   └── training_history.json
├── plots/                               # Plots and attention heatmaps
│   ├── training_curves.png
│   ├── sentence_length_analysis.png
│   ├── attention_sample_*.png
│   └── error_case_*.png
├── data_preprocessing.py                # Part 1: Corpus cleaning, tokenization, vocab
├── models.py                            # Parts 2 & 4: Baseline & Bahdanau Attention models
├── train.py                             # Parts 2, 4, 5: Comparable training pipeline
├── evaluate.py                          # Parts 3 & 8: Test set and length-bucket BLEU evaluation
├── visualize_attention.py               # Parts 6 & 7: Attention heatmaps & error analysis
├── requirements.txt                     # Dependencies list
├── SETUP.md                             # Setup & run documentation
└── REPORT.md                            # Comprehensive technical report
```
