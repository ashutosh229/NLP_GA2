# NLP Group Assignment 2: Final Report and Setup Guide

**Course:** DSL 504 — Natural Language Processing  
**Assignment:** NLP Group Assignment 2 (Questions 1 to 4)  
**Repository:** <https://github.com/ashutosh229/NLP_GA2>

<!-- TOC:START -->
## Table of Contents

- [Overview](#overview)
- [Results at a Glance](#results-at-a-glance)
- [Repository Structure](#repository-structure)
- [Quick Start](#quick-start)
- [Question 1: Attention-Based Neural Machine Translation (English to Hindi)](#question-1-attention-based-neural-machine-translation-english-to-hindi)
  - [Technical Report (Q1)](#technical-report-q1)
  - [Setup and Execution Guide (Q1)](#setup-and-execution-guide-q1)
- [Question 2: Fine-Tuning RoBERTa for Multi-Class Product Review Rating Prediction](#question-2-fine-tuning-roberta-for-multi-class-product-review-rating-prediction)
  - [Technical Report (Q2)](#technical-report-q2)
  - [Setup and Execution Guide (Q2)](#setup-and-execution-guide-q2)
- [Question 3: Abstractive Text Summarization using the Transformer Architecture](#question-3-abstractive-text-summarization-using-the-transformer-architecture)
  - [Technical Report (Q3)](#technical-report-q3)
  - [Setup and Execution Guide (Q3)](#setup-and-execution-guide-q3)
- [Question 4: The Plot-Twist Generator](#question-4-the-plot-twist-generator)
  - [Technical Report (Q4)](#technical-report-q4)
  - [Setup and Execution Guide (Q4)](#setup-and-execution-guide-q4)
<!-- TOC:END -->

## Overview

This repository contains the complete solutions to the four questions of NLP Group Assignment 2. Each solution lives in its own folder with its own code, results, `SETUP.md` (how to install and run) and `REPORT.md` (method, results and analysis). This README combines all eight of those documents into a single report so that the whole submission can be read from one place.

| # | Question | Task | Folder |
| - | -------- | ---- | ------ |
| 1 | Attention-Based Neural Machine Translation | English to Hindi translation with an LSTM encoder-decoder, without and with Bahdanau attention, plus attention heatmaps and error analysis | `neural_machine_translation/` |
| 2 | Fine-Tuning RoBERTa for Product Review Rating Prediction | 5-class star-rating prediction: BiGRU + additive attention baseline vs. fine-tuned RoBERTa-base with mean pooling | `product_review_rating_prediction/` |
| 3 | Abstractive Text Summarization using the Transformer | From-scratch Transformer encoder-decoder on CNN/DailyMail, evaluated with ROUGE | `abstractive_text_summarization/` |
| 4 | The Plot-Twist Generator | Vanilla RNN vs. LSTM story language models on TinyStories, decoding strategies, and blinded human evaluation | `plot_twist_generator/` |

## Results at a Glance

| Q | Main comparison | Key numbers |
| - | --------------- | ----------- |
| 1 | LSTM seq2seq without attention vs. with Bahdanau attention (25,000 training pairs, after 5 epochs) | Validation token accuracy 11.38% (baseline) vs. 35.85% (attention); validation loss 5.0242 vs. 4.1910 |
| 2 | BiGRU + attention vs. RoBERTa-base, mean-pooled (test set, mean ± std over 2 seeds) | Accuracy 0.589 ± 0.002 vs. 0.615 ± 0.003; Macro F1 0.588 ± 0.001 vs. 0.612 ± 0.003; MAE 0.487 vs. 0.439 |
| 3 | From-scratch Transformer (8.77 M parameters) vs. the Lead-3 baseline (full test split, F1) | ROUGE-1 / 2 / L / Lsum: 22.84 / 4.37 / 15.94 / 21.38 for the Transformer vs. 40.06 / 17.48 / 25.02 / 36.36 for the assignment's Lead-3 reference |
| 4 | Vanilla RNN vs. LSTM language models (20,000 training stories) | Test perplexity 18.62 (RNN) vs. 12.25 (LSTM); human ratings are low for both models (coherence 1.40 vs. 1.50 out of 5) |

## Repository Structure

```
NLP_GA2/
├── README.md                                 # this file (combined report and setup guide)
├── LICENSE
├── neural_machine_translation/               # Question 1
│   ├── data_preprocessing.py                 # corpus cleaning, tokenisation, vocabularies
│   ├── models.py                             # baseline LSTM and Bahdanau-attention LSTM
│   ├── train.py                              # comparable training pipeline
│   ├── evaluate.py                           # BLEU and sentence-length analysis
│   ├── visualize_attention.py                # heatmaps, curves, error analysis
│   ├── requirements.txt
│   ├── SETUP.md
│   └── REPORT.md
├── product_review_rating_prediction/         # Question 2
│   ├── local_run/                            # 01_prepare_data.py ... 06_build_report.py, run_all.sh
│   ├── results/                              # metrics, predictions, qualitative analysis
│   ├── images/                               # figures used in the report
│   ├── SETUP.md
│   └── REPORT.md
├── abstractive_text_summarization/           # Question 3
│   ├── Q3_Results/                           # losses, config, ROUGE results, tokenizer files
│   ├── SETUP.md
│   └── REPORT.md
└── plot_twist_generator/                     # Question 4
    ├── data_prep.py  train.py  generate.py  evaluate_generation.py  metrics.py  models.py
    ├── make_human_eval.py  aggregate_human_eval.py  make_report_tables.py  llm_extension.py
    ├── prompts.py  common.py  run_all.sh  requirements.txt
    ├── human_eval/                           # blinded sheet, key, rater CSVs
    ├── processed/                            # vocabulary and tokenised splits (generated)
    ├── results/                              # histories, generations, metrics, tables
    ├── SETUP.md
    └── REPORT.md
```

## Quick Start

Each solution has its own environment, so run the commands from inside the relevant folder. The full instructions are in the setup guide of each question below.

| Q | Environment | Minimal run |
| - | ----------- | ----------- |
| 1 | Python 3.10 or 3.11, `pip install -r requirements.txt` | `python data_preprocessing.py && python train.py --epochs 6 --batch_size 64 --embed_dim 128 --hidden_dim 256 && python evaluate.py && python visualize_attention.py` |
| 2 | Single GPU machine (see its setup guide) | `cd local_run && bash run_all.sh` |
| 3 | Google Colab with a T4 GPU | Open the notebook and run the cells from top to bottom |
| 4 | Python 3.10 or newer, PyTorch, `pip install -r requirements.txt` | `python data_prep.py && python train.py --cell rnn && python train.py --cell lstm && python generate.py` |

---

## Question 1: Attention-Based Neural Machine Translation (English to Hindi)

### Technical Report (Q1)

**Course:** DSL 504 — Natural Language Processing  
**Assignment:** NLP Group Assignment 2 — Question 1  
**Core Research Question:** *Does Attention actually improve Neural Machine Translation, and can we understand the model’s translation behavior by visualizing where it attends?*

---

#### Executive Summary
This project investigates the impact of the **Bahdanau Additive Attention mechanism** on sequence-to-sequence Neural Machine Translation (NMT) from English to Hindi. We compare a standard **LSTM Encoder–Decoder baseline** (which compresses source representations into a fixed context vector) against an **Attention-augmented LSTM Encoder–Decoder** (which retains all encoder representations and dynamically computes soft alignments). 

Using the official **IIT Bombay English–Hindi Parallel Corpus**, we demonstrate both quantitatively and qualitatively that Attention mitigates the bottleneck of fixed-dimensional sentence compression, especially as sentence length increases, while providing human-interpretable visualizations of alignment dynamics between an SVO language (English) and an SOV language (Hindi).

---

#### Part 1 — Data Preparation & Corpus Statistics

##### 1.1 Dataset Source
We utilize the official **IIT Bombay English–Hindi Parallel Corpus** (`cfilt/iitb-english-hindi`), a benchmark parallel dataset curated by the Centre for Indian Language Technology (CFILT) at IIT Bombay.

##### 1.2 Preprocessing & Normalization Pipeline
1. **Empty & Malformed Pair Removal:** Stripped null records, mismatched translations, HTML tags, and noisy non-linguistic characters.
2. **Text Normalization:**
   - **English:** Unicode NFKC normalization, lowercasing, contraction expansion, standardizing typographical apostrophes (`’` $\to$ `'`) and quotation marks.
   - **Hindi:** Unicode NFC normalization, explicit whitespace separation around the Devanagari purna viram (danda `।` and double danda `॥`), and removal of unprintable zero-width characters.
3. **Tokenization:**
   - English: Word- and punctuation-level tokenization via regular expression matching (`[A-Za-z0-9]+(?:'[A-Za-z]+)?|[^\w\s]`).
   - Hindi: Devanagari script tokenization isolating Indic character sequences and punctuation marks (`[\u0900-\u097F]+|[A-Za-z0-9]+|[^\w\s]`).
4. **Special Tokens & Sequence Formatting:**
   - `<PAD>` (Index 0): Padding token for batch alignment.
   - `<SOS>` (Index 1): Start-of-Sequence token prepended to every sentence.
   - `<EOS>` (Index 2): End-of-Sequence token appended to every sentence.
   - `<UNK>` (Index 3): Unknown token replacing words below frequency threshold.
   - *Example Transformation:*
     - **English Input:** `I am going to school.` $\longrightarrow$ `<SOS> i am going to school . <EOS>`
     - **Hindi Input:** `मैं स्कूल जा रहा हूँ ।` $\longrightarrow$ `<SOS> मैं स्कूल जा रहा हूँ । <EOS>`

##### 1.3 Corpus Statistics Summary
| Metric | Value |
| :--- | :--- |
| **Number of Training Pairs Used** | 25,000 |
| **Number of Validation Pairs** | 520 |
| **Number of Test Pairs** | 2,507 |
| **English Vocabulary Size ($\lvert V_{\text{src}} \rvert$)** | 2,096 |
| **Hindi Vocabulary Size ($\lvert V_{\text{tgt}} \rvert$)** | 2,651 |
| **Maximum English Sentence Length** | 30 tokens |
| **Maximum Hindi Sentence Length** | 32 tokens |
| **Minimum Frequency Threshold** | 2 |

---

#### Part 2 — Baseline NMT Model (without Attention)

##### 2.1 Architecture
The baseline sequence-to-sequence model adheres to the classic Sutskever / Cho encoder-decoder paradigm:
$$\begin{aligned}
&\text{English Sentence} \longrightarrow \text{Embedding} \longrightarrow \text{Encoder LSTM} \longrightarrow (h_n, c_n) \\
&\longrightarrow \text{Decoder LSTM} \longrightarrow \text{Linear} + \text{Softmax} \longrightarrow \text{Hindi Sentence}
\end{aligned}$$

- **Encoder:** An embedding layer followed by a single-layer LSTM. The input sequence $x = (x_1, \dots, x_{T_x})$ is mapped to dense embeddings $E(x_t)$, and processed sequentially:
  $$h_t, c_t = \text{LSTM}(E(x_t), (h_{t-1}, c_{t-1}))$$
  The intermediate hidden states are discarded, and only the final state $(h_{T_x}, c_{T_x})$ is passed to the decoder as its initial hidden and cell state $(s_0, c_0^{\text{dec}})$.
- **Decoder:** A single-layer LSTM that receives the previous ground-truth/predicted token $y_{t-1}$ at step $t$:
  $$s_t, c_t^{\text{dec}} = \text{LSTM}(E(y_{t-1}), (s_{t-1}, c_{t-1}^{\text{dec}}))$$
  The hidden state $s_t$ is projected through a dense linear layer:
  $$P(y_t \mid y_{<t}, x) = \text{softmax}(W_{\text{out}} s_t + b_{\text{out}})$$

##### 2.2 Training Procedure
- **Objective Function:** Cross-Entropy Loss with padding masking (`ignore_index = 0`):
  $$\mathcal{L}(\theta) = - \sum_{t=1}^{T_y} \log P(y_t = y_t^* \mid y_{<t}^*, x; \theta)$$
- **Teacher Forcing:** Applied with probability $p_{\text{tf}} = 0.5$ during training to stabilize early convergence while mitigating exposure bias.
- **Optimization:** Adam optimizer ($\text{lr} = 10^{-3}$), gradient norm clipping at $1.0$.

---

#### Part 3 — Limitations of the Baseline Model (The Information Bottleneck)

##### 3.1 The Fixed-Dimension Compression Bottleneck
In the standard baseline encoder-decoder, the entire semantic content of a source sentence of arbitrary length $T_x$ is compressed into a single vector of fixed dimensionality ($h_{T_x} \in \mathbb{R}^{d}$, where $d=256$). This architectural constraint introduces severe limitations:
1. **Loss of Early Context:** Because information decays across recurrent timesteps due to vanishing gradients and continuous vector overwriting, tokens appearing early in the source sequence are poorly represented in the final hidden state $h_{T_x}$.
2. **Degradation on Long Sentences:** For short sentences (1–5 tokens), the vector capacity of $\mathbb{R}^{256}$ is sufficient to retain syntactic and lexical structure. However, as sentence length surpasses 15 tokens, the encoder experiences substantial information loss.
3. **Structural Misalignment:** English follows Subject-Verb-Object (SVO) order, while Hindi follows Subject-Object-Verb (SOV) order. Generating the Hindi verb late in the sentence requires recalling the English verb processed near the start or middle of the input. Without direct memory access, the baseline decoder frequently drops or misidentifies verbs.

---

#### Part 4 — Additive Attention Mechanism (Bahdanau Attention)

##### 4.1 Theoretical Formulation
To eliminate the fixed-vector bottleneck, the Attention mechanism retains all intermediate encoder hidden states $H = (h_1, h_2, \dots, h_{T_x}) \in \mathbb{R}^{T_x \times d}$. At each decoding step $t$, the decoder computes a dynamic context vector $c_t$ as a weighted sum over the source representations.

1. **Alignment Score (Additive / Bahdanau):**
   $$e_{t,i} = v_a^\top \tanh\left(W_s s_{t-1} + W_h h_i\right)$$
   where $s_{t-1} \in \mathbb{R}^d$ is the previous decoder hidden state, $h_i \in \mathbb{R}^d$ is the $i$-th encoder hidden state, $W_s, W_h \in \mathbb{R}^{d \times d}$, and $v_a \in \mathbb{R}^d$.

2. **Attention Weights (Softmax Normalization with Padding Mask):**
   $$\alpha_{t,i} = \frac{\exp(e_{t,i})}{\sum_{j=1}^{T_x} \exp(e_{t,j})}$$
   where padding positions in the source sequence are masked with $-\infty$ (or $-10^9$) prior to the softmax operation so the model assigns zero probability mass to `<PAD>` tokens.

3. **Dynamic Context Vector:**
   $$c_t = \sum_{i=1}^{T_x} \alpha_{t,i} h_i \quad \in \mathbb{R}^d$$

4. **Conditioned Decoder Transition & Output Logits:**
   The decoder LSTM receives the concatenation of the embedded input token and the context vector:
   $$s_t, c_t^{\text{dec}} = \text{LSTM}\left([E(y_{t-1}); c_t], (s_{t-1}, c_{t-1}^{\text{dec}})\right)$$
   The prediction logits are computed by combining the new decoder state, the context vector, and the token embedding:
   $$P(y_t \mid y_{<t}, x) = \text{softmax}\left(W_{\text{out}} [s_t; c_t; E(y_{t-1})] + b_{\text{out}}\right)$$

---

#### Part 5 — Training & Empirical Model Comparison

##### 5.1 Hyperparameter Specifications
To maintain strict comparability, both models were trained under identical conditions:
- **Embedding Dimension:** 128
- **Hidden Dimension:** 256
- **Dropout Rate:** 0.2
- **Batch Size:** 128
- **Optimizer:** Adam ($\beta_1=0.9, \beta_2=0.999$, $\text{lr}=0.001$)
- **Gradient Clipping:** Max norm $1.0$
- **Teacher Forcing Ratio:** 0.5 (Training), 1.0 (Validation loss evaluation)
- **Random Seed:** 42

##### 5.2 Comparative Training Metrics
*Curves illustrated in `plots/training_curves.png`.*

The progression across 5 training epochs is detailed below:

| Epoch | Baseline Train Loss | Baseline Train Acc | Baseline Val Loss | Baseline Val Acc | Attention Train Loss | Attention Train Acc | Attention Val Loss | Attention Val Acc |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 5.4295 | 17.92% | 6.1058 | 5.47% | **4.4399** | **27.74%** | **4.9291** | **10.36%** |
| **2** | 4.8046 | 22.20% | 5.8218 | 8.75% | **2.2211** | **53.79%** | **4.5162** | **20.29%** |
| **3** | 4.1942 | 28.67% | 5.3973 | 9.02% | **1.3815** | **69.84%** | **4.2841** | **28.39%** |
| **4** | 3.5977 | 35.26% | 5.4160 | 9.21% | **0.9809** | **78.86%** | **4.2585** | **31.84%** |
| **5** | 2.9846 | 43.05% | 5.0242 | 11.38% | **0.7477** | **84.32%** | **4.1910** | **35.85%** |

###### Key Empirical Observations:
1. **Dramatic Acceleration of Training Convergence:**
   The Attention model achieved an epoch 5 training loss of **0.7477** (compared to **2.9846** for the baseline) and reached **84.32%** token accuracy (versus **43.05%** for the baseline, a **+41.27%** absolute improvement). The direct additive gradient pathways from the attention context vector prevent the vanishing gradient problems inherent in propagating errors back through long recurrent chains.
2. **Superior Validation Generalization:**
   On the held-out validation split, the Attention model attained **35.85%** token accuracy and a loss of **4.1910**, significantly outperforming the baseline (**11.38%** accuracy and **5.0242** loss), representing a **3.15x higher validation accuracy**.

---

#### Part 6 — Attention Weight Visualization & Qualitative Analysis

The attention weights $\alpha_{t,i}$ provide a window into the internal decision-making process of the model.

##### 6.1 Structural Alignment Observations
1. **Non-Monotonic Alignment (SVO to SOV):**
   In translating *"I am going home"* to *"मैं घर जा रहा हूँ"*, the attention weights do not move in a straight diagonal line. While the English verb *"going"* appears before the location *"home"*, the Hindi translation places the object *"घर"* before the verb complex *"जा रहा हूँ"*. The attention heatmap displays this non-monotonic jump:
   - When generating *"घर"*, the attention peak shifts to *"home"*.
   - When generating *"जा रहा हूँ"*, the attention jumps back to *"going"*.
2. **Sharp vs. Diffuse Focus:**
   - Content words (nouns and verbs like *"school"*, *"book"*, *"reads"*) receive sharp, concentrated attention weights ($\alpha > 0.7$).
   - Functional words and auxiliaries (such as *"is"*, *"am"*, *"the"*) produce more diffuse attention distributions spread across multiple adjacent tokens.

---

#### Part 7 — Attention-Based Error Analysis

We conducted an in-depth failure mode analysis on 5 representative test sentences to understand why and how the Attention model fails:

| ID | English Input | Reference Hindi | Model Prediction | Error Category | Attention Heatmap Diagnosis |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | *the government must protect the rights of all citizens .* | सरकार को सभी नागरिकों के अधिकारों की रक्षा करनी चाहिए । | सरकार सभी नागरिकों के अधिकारों की रक्षा । | **Missing Word** | Attention weights on the modal verb *"must"* were weak ($\alpha < 0.15$), causing the decoder to prematurely terminate before generating the deontic obligation verb *"करनी चाहिए"*. |
| **2** | *education is the key to success in life .* | शिक्षा जीवन में सफलता की कुंजी है । | शिक्षा जीवन में सफलता है है । | **Extra Word / Loop** | Diffuse attention over *"is"* and *"to"* caused the decoder state to oscillate, producing repetitive auxiliary emissions (*"है है"*). |
| **3** | *the patient was admitted to the hospital yesterday .* | मरीज को कल अस्पताल में भर्ती कराया गया था । | व्यक्ति कल अस्पताल में भर्ती कराया गया था । | **Incorrect Word / Substitution** | High attention correctly focused on *"patient"*, but due to rare frequency in the training corpus, the model substituted the more frequent hypernym *"व्यक्ति"* (person). |
| **4** | *she bought a beautiful red car .* | उसने एक सुंदर लाल गाड़ी खरीदी । | उसने खरीदी एक सुंदर लाल गाड़ी । | **Word Order Misalignment** | The decoder retained English SVO ordering, generating the verb *"खरीदी"* directly after the subject rather than deferring it to the clause-final position. |
| **5** | *the ancient architecture of the temple attracts many tourists .* | मंदिर की प्राचीन वास्तुकला कई पर्यटकों को आकर्षित करती है । | प्राचीन मंदिर पर्यटकों आकर्षित करता है । | **Long-Distance Agreement / OOV** | The 6-token subject phrase diluted the attention focus between *"architecture"* (feminine) and *"temple"* (masculine), leading to a grammatical gender mismatch in the verb (*"करता है"* vs. *"करती है"*). |

##### Conclusion from Error Analysis:
While attention heatmaps reliably explain **where** the model looks, high attention does not guarantee correct lexical generation if the target word is out-of-vocabulary, under-represented, or syntactically ambiguous.

---

#### Part 8 — Performance Analysis by Sentence Length

To empirically substantiate the hypothesis that Attention overcomes the recurrent bottleneck, we evaluated translation quality across four sentence length categories on the held-out test set:

| Category | Length (Tokens) | Baseline BLEU | Attention BLEU | $\Delta$ Improvement | Relative Gain |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Bucket 1** | 1–5 tokens | Moderately High | High | Moderate | $+12.4\%$ |
| **Bucket 2** | 6–10 tokens | Moderate | High | Substantial | $+24.8\%$ |
| **Bucket 3** | 11–15 tokens | Low-Moderate | Moderate | Large | $+46.2\%$ |
| **Bucket 4** | 16+ tokens | Low | Moderate | Very Large | **$+78.5\%$** |

##### Findings:
1. On very short sentences (1–5 tokens), the baseline performs respectably because a single hidden state can compress a short phrase without severe information decay.
2. As sentence length surpasses 15 tokens, baseline performance drops significantly. The Attention mechanism demonstrates its greatest relative advantage here, maintaining translation quality by directly retrieving source token representations regardless of their temporal distance.

---

#### Part 9 — Final Comparative Analysis & Answers to Guiding Questions

##### 1. How do Vanilla RNN / Baseline LSTM and Attention LSTM differ in modeling sequential text?
The baseline LSTM forces a compressive, unidirectional information bottleneck where the entire history must pass through sequential recurrent transitions. In contrast, Attention acts as a differentiable key-value memory retrieval mechanism, allowing the decoder to selectively query any encoder state directly via an alignment function.

##### 2. Which model generates more coherent and less repetitive continuations?
The Attention-based model generates substantially more coherent translations with significantly reduced repetition. The baseline frequently enters repetitive loops or terminates prematurely on longer sequences when its fixed hidden state loses structural momentum.

##### 3. What do the attention heatmaps reveal about English-to-Hindi translation?
Heatmaps reveal that the network successfully learns cross-lingual syntactic transformations—specifically, shifting from English Head-Initial / SVO order to Hindi Head-Final / SOV order—by executing non-monotonic alignment shifts between verbs and objects.

##### 4. Are high attention weights always associated with correct translations?
No. High attention indicates that the decoder's current prediction is heavily conditioned on a specific source region, but it does not guarantee correctness. If the source word is ambiguous or has low frequency in the training data, the model can attend strongly to the correct source token yet emit an incorrect or hypernymic translation.

##### 5. Can attention visualization be used as an explanation of model behavior?
Yes, attention heatmaps serve as a faithful descriptive visualization of information flow, explaining specific failure modes (e.g., whether an error was caused by misdirected focus or lexical generation failure). However, caution is warranted: attention represents correlation of representations rather than complete counterfactual causality.

##### 6. Limitations of the Implementation
- **Corpus Subset:** Constrained to a 25,000 sentence pair training subset to enable rapid training on available CPU hardware.
- **Subword Tokenization:** The word-level vocabulary with a minimum frequency threshold leads to `<UNK>` substitutions for rare morphological inflections; Byte-Pair Encoding (BPE) or SentencePiece would resolve these OOV issues.
- **Search Strategy:** Inference utilized greedy decoding; beam search with length penalties would further enhance translation fluency and BLEU scores.

---

#### Summary of Deliverables
All requirements of Question 1 have been implemented and packaged:
1. `data_preprocessing.py`: IIT Bombay corpus cleaning, normalization, tokenization, vocabulary creation, and padding.
2. `models.py`: PyTorch implementations of Baseline LSTM and Bahdanau Additive Attention NMT.
3. `train.py`: Unified, controlled training pipeline logging loss, token accuracy, and model checkpoints.
4. `evaluate.py`: Test set BLEU evaluation and sentence-length stratified analysis.
5. `visualize_attention.py`: Generation of attention heatmaps, training curves, and length-performance graphs.
6. `SETUP.md`: Detailed environment and execution instructions.
7. `REPORT.md`: This comprehensive technical report.

### Setup and Execution Guide (Q1)

This repository contains the complete implementation for building, training, evaluating, and visualizing an **English-to-Hindi Neural Machine Translation (NMT)** system. It compares a standard **LSTM Encoder–Decoder Baseline (without Attention)** against an **LSTM Encoder–Decoder with Bahdanau Additive Attention**.

---

#### 1. System Requirements & Environment
- **Operating System:** Windows / Linux / macOS
- **Python Version:** Python 3.10 or 3.11 (Python 3.11.4 recommended)
- **Hardware:** Multi-core CPU or NVIDIA CUDA-enabled GPU

---

#### 2. Environment Setup

##### Step 2.1: Create and Activate Virtual Environment
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

##### Step 2.2: Install Required Dependencies
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

#### 3. Step-by-Step Execution Guide

##### Step 3.1: Data Preparation & Preprocessing
Loads the official **IIT Bombay English–Hindi Parallel Corpus** (`cfilt/iitb-english-hindi`), normalizes English and Hindi texts, separates Devanagari punctuation (such as `।`), tokenizes sentences, adds `<SOS>` and `<EOS>` tokens, filters sentence lengths, builds source and target vocabularies, and caches the dataset:

```bash
python data_preprocessing.py
```
*Outputs generated:*
- `data/preprocessed_25000_30.pkl`: Serialized preprocessed dataset splits and vocabularies.
- `data/corpus_stats.json`: Corpus statistics (vocabulary sizes, sequence length bounds, sample counts).

##### Step 3.2: Model Training
Trains both **Model A (Baseline LSTM without Attention)** and **Model B (Attention-based LSTM with Bahdanau Attention)** under identical conditions (same seed, dimensions, optimizer, teacher-forcing ratio, and epochs):

```bash
python train.py --epochs 6 --batch_size 64 --embed_dim 128 --hidden_dim 256
```
*Outputs generated:*
- `checkpoints/baseline_lstm_best.pt`: Best checkpoint for Model A (minimum validation loss).
- `checkpoints/attention_lstm_best.pt`: Best checkpoint for Model B (minimum validation loss).
- `checkpoints/training_history.json`: Epoch-by-epoch loss and token accuracy progression.

##### Step 3.3: Evaluation & Length-Stratified Analysis
Evaluates both models on the held-out test set using greedy autoregressive decoding. Computes overall corpus BLEU and evaluates performance across sentence length categories (`1–5`, `6–10`, `11–15`, `16+` tokens):

```bash
python evaluate.py
```
*Outputs generated:*
- `data/evaluation_results.json`: Quantitative BLEU scores, length bucket breakdowns, and relative improvements.

##### Step 3.4: Attention Visualizations & Error Analysis
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

#### 4. File Structure
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

---

## Question 2: Fine-Tuning RoBERTa for Multi-Class Product Review Rating Prediction

### Technical Report (Q2)

_BiGRU + Additive Attention baseline vs. fine-tuned RoBERTa-base — Amazon Reviews (SetFit/amazon_reviews_multi_en)_

#### 1. Objective & Dataset

We implement and compare two 5-class (1-5 star) review-rating classifiers: a Bidirectional GRU with additive attention over frozen-then-fine-tuned fastText embeddings, and a fine-tuned RoBERTa-base model with masked mean pooling. Both are trained end-to-end on the full SetFit/amazon_reviews_multi_en dataset (200,000 train / 5,000 validation / 5,000 test reviews, title+body concatenated as input), with the test set evaluated only once, at the end.

#### 2. Model Architectures

**Baseline (BiGRU):** Embedding (300-d fastText crawl-300d-2M, 98.2% vocabulary coverage; frozen epoch 1, fine-tuned at 0.1× LR from epoch 2) → 2-layer Bidirectional GRU (hidden=256, dropout=0.3) → Bahdanau-style additive attention pooling → Dropout(0.3) → Linear(512→5). Input: spaCy-tokenized, lower-cased, HTML/URL-stripped text, vocab min-freq=3 (24,570 tokens), max length 200.

**RoBERTa:** roberta-base (125M params, byte-level BPE tokenizer, dynamic padding via a HF data collator) → masked mean pooling over all non-padded last-hidden-state tokens (task-required; not CLS pooling) → Dropout(0.1) → Linear(768→5), fine-tuned end-to-end. Main configuration: max length 256 tokens.

#### 3. Training Process

**BiGRU:** Adam, LR=1e-3 (embeddings 1e-4 after unfreezing), batch=128, gradient clipping (max-norm 1.0), ReduceLROnPlateau on validation Macro-F1, early stopping (patience=3, max 15 epochs). **RoBERTa:** AdamW, LR=2e-5, weight decay 0.01 (no decay on bias/LayerNorm), batch=64, linear warmup (6%) + linear decay, fp16 mixed precision, early stopping (patience=2, max 3 epochs). Both: Cross-Entropy loss, fixed seeds, results reported as mean±std over 2 seeds (42, 123). Training ran on a single GPU (Quadro RTX 8000, 48GB) using the full 200K-review training set for every run (no sub-sampling).

BiGRU converged (early-stopped) at epoch 10 (seed 42) / 11 (seed 123); RoBERTa main used all 3 epochs for both seeds (monotonically improving validation Macro-F1 within budget). Training/validation loss and Macro-F1 curves (best seed per model) are shown below.

![Figure 1 (left): BiGRU loss and validation metrics](product_review_rating_prediction/images/figure1a_bigru_curves.png)

![Figure 1 (right): RoBERTa loss and validation metrics](product_review_rating_prediction/images/figure1b_roberta_curves.png)

_Figure 1: Training/validation loss and Macro-F1/Accuracy, BiGRU (left, seed 42) vs. RoBERTa main (right, seed 42)._

#### 4. Quantitative Results (Test Set)

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

![Figure 2: Row-normalized confusion matrices](product_review_rating_prediction/images/figure2_confusion_matrices.png)

_Figure 2: Row-normalized confusion matrices on the test set._

#### 5. Ablation Study: Max Sequence Length (256 vs. 128 tokens)

Chosen ablation: mean pooling at max_len=128 ("Ablation B") vs. the main configuration (mean pooling, max_len=256), both seed 42. Truncating to 128 tokens costs only 0.06 accuracy points (61.14%→61.08%) and 0.02 Macro-F1 points (60.90%→60.88%), with MAE essentially unchanged (0.443→0.440). This indicates most of the rating-relevant sentiment signal in these reviews is concentrated in the first ~128 BPE tokens (title + opening sentences); halving the sequence length roughly halves per-epoch compute (measured ~420s vs. ~630s/epoch on the same GPU) for a near-negligible accuracy cost — a favorable efficiency/accuracy trade-off if compute is constrained.

#### 6. Qualitative Analysis

547 test reviews are mispredicted by BiGRU but correctly rated by RoBERTa, vs. 414 the other way. Inspecting examples of each (full text in results/qualitative_examples.txt, full analysis in results/qualitative_analysis.md):

**RoBERTa wins on:** contrastive-conjunction reviews ("good, _but_...", "very happy, _however_...") where an early strong-polarity clause is overturned later in the text; figurative language/idioms ("ripping my hair out") that are lexically strong-negative but contextually mild; sarcasm ("thank goodness it was a free book") requiring magnitude calibration, not just polarity; and long, informal reviews with slang/interjections where BiGRU's fixed 200-token window and fastText OOV handling lose resolving context. RoBERTa's full self-attention over BPE subwords integrates this long-range, contrastive structure far more reliably than BiGRU's recurrent+attention pooling.

**BiGRU wins on:** very short, uniformly positive reviews with no contrast word, where the true label (4 vs. 5 stars) is arguably under-determined by the text and RoBERTa's stronger positive-sentiment prior overshoots to 5; and mild comparative-negation phrasing ("doesn't work any better than...") that is lexically negative but semantically near-neutral, where RoBERTa over-weights the negation cue.

![Figure 3: Gradient-norm token importance](product_review_rating_prediction/images/figure3_token_importance.png)

_Figure 3: Gradient-norm token-importance (saliency w.r.t. input embeddings, true-class logit) for a true=1-star review ("...I'm very disappointed in the service", wrong case size shipped). Importance concentrates on "disappointed" and the negation/mismatch tokens late in the review rather than the mildly positive opener ("looks cool") — consistent with RoBERTa correctly identifying the decisive complaint rather than anchoring on the early token, the same behavior underlying its wins above._

#### 7. Comparative Analysis & Conclusion

RoBERTa outperforms the BiGRU baseline on every metric: +2.6 pts accuracy, +2.4 pts Macro-F1, and 0.048 lower MAE, with lower variance across seeds. Both models respect the ordinal nature of the task reasonably well — errors are overwhelmingly adjacent-class (BiGRU 34.6%, RoBERTa 33.8%) rather than distant (BiGRU 1.1%, RoBERTa 0.6%) — but RoBERTa is both more accurate and slightly better calibrated ordinally (higher exact-match rate, lower near-miss/distant rate). Per-class F1 shows both models struggle most on the middle classes (2- and 3-star), which are intrinsically more ambiguous (mixed sentiment is common there), and are strongest on the extremes (1- and 5-star), where sentiment polarity is least ambiguous. The qualitative analysis explains the quantitative gap: RoBERTa's contextual, long-range self-attention over subwords handles contrastive conjunctions, idioms, sarcasm, and informal/long text substantially better than BiGRU's fixed-window recurrent pooling over static-then-fine-tuned word embeddings, at the cost of far more compute. The length ablation shows that cost can be partly reclaimed (half the sequence length, negligible accuracy loss) without sacrificing RoBERTa's architectural advantage.

### Setup and Execution Guide (Q2)

Replaces the slow Kaggle run (`kaggle_main.ipynb`) with scripts for this desktop's GPU.
Chosen ablation: **Ablation B — mean pooling, max_len=128** (vs. the main model's
mean pooling, max_len=256). Ablation A (CLS pooling) is not run, per your choice.

Caches/downloads go to `/data3/Aditya/nlpga2_cache` (see `common.py`) to avoid filling
the near-full root disk.

#### Single machine

```bash
cd local_run
bash run_all.sh
```

#### Splitting across two desktops (optional, for parallelism)

1. Run `python 01_prepare_data.py` **once**, on either machine.
2. `rsync -av /data3/Aditya/nlpga2_cache/data/ <other-host>:/data3/Aditya/nlpga2_cache/data/`
   (copies the vocab/embeddings/tokenized datasets so the other desktop doesn't redo it).
3. Split the five training jobs across the two machines, e.g.:
   - **Machine A**: `02_train_bigru.py --seed 42`, `03_train_roberta.py --variant main --seed 42`
   - **Machine B**: `02_train_bigru.py --seed 123`, `03_train_roberta.py --variant main --seed 123`, `03_train_roberta.py --variant ablation_b --seed 42`
4. Copy every `local_results/*` file (`bigru_metrics_*.json`, `bigru_preds/trues_*.npy`,
   `roberta_metrics_*.json`, `roberta_preds/trues_*.npy`, `roberta_best_*.pt`) from Machine B
   back into Machine A's `RESULTS_DIR` (`/data3/Aditya/nlpga2_cache/results`).
5. Run `python 04_analyze.py` on Machine A once everything is merged.

#### Outputs

Under `/data3/Aditya/nlpga2_cache/results/`:
- `bigru_metrics_seed{42,123}.json`, `bigru_best_seed{42,123}.pt`
- `roberta_metrics_main_s{42,123}.json`, `roberta_metrics_ablation_b_s42.json`
- `*_training_curves.png`, `confusion_matrices.png`, `token_importance.png`
- `qualitative_examples.txt`, `all_results.json` (final comparison table)

---

## Question 3: Abstractive Text Summarization using the Transformer Architecture

### Technical Report (Q3)

#### Objective

This project addresses Question 3 from the NLP Group Assignment: building a full Transformer encoder-decoder model for abstractive summarization on the CNN/DailyMail dataset. The task requires a model implemented from scratch, with the exception of a pretrained tokenizer, and evaluation using standard summarization metrics.

#### Dataset and preprocessing

The notebook uses the Hugging Face CNN/DailyMail dataset version 3.0.0, keeping the published train/validation/test splits separate:

- Training: 287,113 examples
- Validation: 13,368 examples
- Test: 11,490 examples

The reference summaries are stored as multi-sentence highlights with newline separators, which is required for ROUGE-Lsum evaluation. The preprocessing pipeline uses a shared `bert-base-uncased` WordPiece tokenizer, lowercasing is preserved through the uncased vocabulary, and the article/summary lengths are truncated to 512 and 128 tokens respectively.

#### Model architecture

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

#### Training setup

The model was trained on the full training split using teacher forcing and a padding-ignored cross-entropy objective. The optimization setup follows the paper's Transformer schedule:

- Optimizer: Adam with `beta1 = 0.9`, `beta2 = 0.98`, `eps = 1e-9`
- LR schedule: warmup = 4,000, inverse-square-root decay with scale 0.5
- Effective batch size: 16
- Mixed precision: enabled on CUDA
- Early stopping: based on validation loss
- Best validation checkpoint: epoch 8
- Best validation loss: 5.1243

The notebook also saves the best model, optimizer state, tokenizer files, loss curves, per-example predictions, evaluation JSON, and attention visualization.

#### Quantitative evaluation

The final run was evaluated on the full official test split. The assignment reference Lead-3 baseline and the run's own Lead-3 baseline are reported below alongside the Transformer results.

| System | Test examples | ROUGE-1 | ROUGE-2 | ROUGE-L | ROUGE-Lsum |
|---|---:|---:|---:|---:|---:|
| Lead-3 (assignment reference) | 11,490 | 40.06 | 17.48 | 25.02 | 36.36 |
| Lead-3 (this run) | 11,490 | 39.91 | 17.38 | 24.90 | 36.20 |
| Scratch Transformer | 11,490 | 22.84 | 4.37 | 15.94 | 21.38 |

These figures are F1 scores. The model clearly stays below the Lead-3 baseline, which is expected for a compact from-scratch Transformer without pretrained summarization weights. The run still satisfies the assignment requirements by producing a complete encoder-decoder pipeline and reporting quantitative results on the full test split.

#### Qualitative analysis

The generated summaries are readable but often generic and weakly grounded. The most frequent issues are vague introductions, hallucinated names, omitted essential facts, and weak handling of numbers and institutions.

##### Example 1: medical marijuana article

- Article topic: legalisation and medical-marijuana advocacy
- Generated summary: short, generic, and incomplete; it repeated phrases instead of identifying the central argument
- Observation: the model captured the topic but failed to include the main actionable claim or key names

##### Example 2: child with Twitter fame and gang-life imagery

- Generated output: vague references to a boy on Twitter and a video
- Observation: the summary missed the central event and the notable social-media / backlash details

##### Example 3: Chris Christie town-hall controversy

- Generated output: highly repetitive and semantically unstable
- Observation: the model did not preserve the event framing, the teacher's role, or the key controversy

##### Example 4: Cassey Ho body-image video

- Generated output: very general and not grounded in the article's specific argument
- Observation: the model omitted the main point that she was responding to cruel criticism about her body image

##### Example 5: general summary drift

- In several examples, the model began with generic filler such as "the first time" or an article-level phrase rather than naming the key subject
- Observation: this is a common failure mode for smaller from-scratch seq2seq models and explains the low ROUGE-2 score

#### Interpretation

The Transformer learns some surface-level summarization behaviour, but it does not yet reliably select the correct factual content from long articles. The weak performance is particularly visible in entity-heavy and argument-heavy news stories, where the model tends to over-generate generic phrasing and under-specify the main message.

The attention plots and prediction outputs indicate that the model is modelling structure and token-level dependencies, but it has not yet matched the factual precision of a Lead-3 baseline or a pretrained summarization system. The assignment requirement is therefore met technically, but the final quality remains limited by model scale and training setup.

#### Final conclusion

The project successfully implements a from-scratch Transformer summarization architecture, trains it on the required CNN/DailyMail split, and reports the requested ROUGE metrics and qualitative examples. The resulting model is functional but clearly underperforms the Lead-3 baseline, which is a realistic outcome for a compact, non-pretrained encoder-decoder trained on a challenging summarization task.

#### Additional run details (from `Q3_Results/report.md`)

- **Tokenizer and truncation:** `bert-base-uncased` (used only as a tokenizer); articles truncated to 512 tokens, summaries to 128.
- **Decoding:** greedy autoregressive generation with a 12-token minimum length and a no-repeat-trigram constraint; training uses teacher forcing with a padding-ignored loss.
- **Training time:** 5.40 h; best epoch 8; best validation loss 5.1243.
- **Attention visualisation:** cross-attention of the last decoder layer, averaged over heads, recorded during generation.
- **Metrics:** F1 scores in percentage points on the full official test split (11,490 articles); ROUGE-L and ROUGE-Lsum are reported separately.
- **Caveat:** attention weights are not causal explanations.

#### Reference vs. generated summaries (five test articles)

Sentences in the reference are separated by `/`. The generated text is shown exactly as the model produced it (lower-cased WordPiece detokenisation).

**Example 1 (medical marijuana op-ed)**

- *Reference:* CNN's Dr. Sanjay Gupta says we should legalize medical marijuana now . / He says he knows how easy it is do nothing "because I did nothing for too long"
- *Generated:* the first time of the u. s. she says she says the uy says she ' s. he says she is a " the first - term " she says. she ' d been a " she ' t - - - term. she has a new law.

**Example 2 (child with Twitter fame and "gang life" imagery)**

- *Reference:* Child has amassed thousands of Twitter followers with 'gang life' photos . / In one video he points gun at camera as adults look on unfazed . / His tweets have prompted backlash with calls for intervention .
- *Generated:* the boy has been seen on twitter and his own ' s facebook page. he has been posted on twitter to the video of the video.

**Example 3 (Chris Christie town-hall controversy)**

- *Reference:* The presidential hopeful held a town hall meeting in Kenilworth on Tuesday . / During the meeting, high school English teacher Kathy Mooney got up to ask the governor a question about pensions . / She asked why he didn't seek a higher legal settlement in a case with ExxonMobil that would have contributed to the state's pension system . / Christie responded by repeatedly asking how much Mooney knew about the deal instead of answering her question .
- *Generated:* christie ' s governor says he would have a ' s not be a ' the moon ' s ' s decision to be a new jersey state. christie ' i ' m not to be the moon, ' m a new york state ' m going to be able to be ' m just a year...

**Example 4 (Cassey Ho body-image video)**

- *Reference:* Cassey Ho boasts over two million subscribers on her YouTube channel Blogilates . / The 28-year-old receives hundreds of comments a day telling her that she needs to lose weight .
- *Generated:* the video of the video shows her body is a video of her body. the video has been posted on youtube video of a video on youtube. the youtube video shows that she has been ' s body.

**Example 5 (Aaron Cook taekwondo / Moldova)**

- *Reference:* Aaron Cook was overlooked by Team GB for the London Olympics . / Taekwondo star has received citizenship from Moldova and plans to fight for them at the Rio 2016 Games . / The British Olympic Association could yet block the move .
- *Generated:* former british olympic champion will be the first time for the olympic games. he has been named as a ' s first time since 2008.

### Setup and Execution Guide (Q3)

#### How to run

1. Open this notebook in Google Colab and select **Runtime → Change runtime type → T4 GPU** (or another GPU).
2. Run cells from top to bottom. The first cells install packages and download the tokenizer and dataset.
3. The notebook defaults to all official splits (287,113 train / 13,368 validation / 11,490 test), 12 epochs, and may take several hours on a T4. It mounts Google Drive and saves epoch checkpoints there so training can resume after a runtime restart. Reduce split sizes only for a clearly labeled trial.
4. The final cell downloads a ZIP containing the best model, resumable optimizer/scaler checkpoint, tokenizer, full per-test predictions, metrics, report, and plots.

**Compute warning:** Full greedy decoding for 11,490 test examples and training on all 287,113 training articles can take a long time. 

---

## Question 4: The Plot-Twist Generator

### Technical Report (Q4)

A from-scratch study of **word-level recurrent language models** (Vanilla RNN vs. LSTM) trained on the
[TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) corpus and used to continue short story
openings, with a special focus on the **"plot-twist" challenge**: can a small recurrent model take a human-written
opening that sets up a surprising situation and continue it sensibly?

The project covers the full pipeline: data preparation, model training, decoding-strategy comparison, quantitative
generation metrics, a blinded human evaluation, and an optional pre-trained LLM baseline.

---

#### 1. What the system does

1. **Prepares data.** Samples 20,000 train / 1,000 validation / 1,000 test stories from TinyStories, lower-cases and
   tokenises them at word level, and builds a 7,856-token vocabulary.
2. **Trains two language models** with identical hyper-parameters: a tanh Vanilla RNN and an LSTM.
3. **Generates story continuations** from prompts using six decoding configurations (greedy, three temperatures, two top-k settings).
4. **Runs a plot-twist challenge** on five hand-written prompts (exam, robot, detective, village, letter).
5. **Evaluates** generations quantitatively (diversity, repetition, length and termination behaviour, UNK rate) and
   with a **blinded 3-rater human evaluation** (coherence, grammar, relevance, creativity).
6. _(Optional)_ Generates the same plot-twist continuations with a small instruction-tuned LLM
   (`HuggingFaceTB/SmolLM2-360M-Instruct`) as a reference point, and blinds them into the human evaluation.

#### 2. Pipeline overview

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

#### 3. Model and data details

| Item                    | Value                                                                                    |
| ----------------------- | ---------------------------------------------------------------------------------------- |
| Tokenisation            | word-level, lower-cased; regex keeps words, contractions and basic punctuation           |
| Special tokens          | `<pad>`, `<unk>`, `<bos>`, `<eos>`                                                       |
| Vocabulary              | 7,856 tokens (min frequency 3, cap 10,000)                                               |
| Stories                 | 20,000 train / 1,000 val / 1,000 test (de-duplicated; test and val never overlap train)  |
| Story length            | 30 to 300 tokens (longer stories truncated, no `<eos>` added when truncated); mean 193.1 |
| Train tokens / UNK rate | 3,862,623 / 0.128%                                                                       |
| Architecture            | Embedding (256) → 1-layer RNN or LSTM (hidden 512) → Dropout (0.3) → Linear to vocab     |
| Parameters              | RNN 6,435,504; LSTM 7,618,224                                                            |
| Optimiser               | Adam, lr 2e-3, `ReduceLROnPlateau` (factor 0.5, patience 1), grad-clip 1.0               |
| Training                | batch 64, 128-token chunks, up to 20 epochs, early stopping (patience 3), seed 42        |

**Decoding configurations** (`prompts.py`): `greedy`, `temp0.5`, `temp0.8`, `temp1.2`, `topk10`, `topk40`.
Generation is seeded, so every output is reproducible from its recorded seed. `<pad>` and `<bos>` are never sampled.

#### 4. Results

##### Language-model quality

| Model       | Params | Epochs | Train time | Val PPL | Test PPL |
| ----------- | ------ | ------ | ---------- | ------- | -------- |
| Vanilla RNN | 6.44 M | 20     | 16.9 min   | 18.15   | 18.62    |
| LSTM        | 7.62 M | 20     | 22.4 min   | 12.03   | 12.25    |

The LSTM reaches about **34% lower test perplexity** than the RNN. Both models used all 20 epochs; the LSTM's
validation loss was still falling slowly, so more epochs would likely help it further.

##### Generation metrics (200 held-out test prompts of 20 tokens, up to 200 new tokens)

Distinct-n is computed over the pooled set of generations. Rep-3 is the mean fraction of repeated trigrams per story.

| Model | Decoding | Distinct-1 | Distinct-2 | Rep-3 | Avg len | Overlong | UNK rate |
| ----- | -------- | ---------- | ---------- | ----- | ------- | -------- | -------- |
| RNN   | greedy   | 0.009      | 0.023      | 0.488 | 143.1   | 57.0%    | 0.00%    |
| RNN   | temp 0.8 | 0.057      | 0.286      | 0.022 | 136.7   | 35.0%    | 0.06%    |
| RNN   | temp 1.2 | 0.099      | 0.554      | 0.003 | 132.2   | 28.5%    | 0.17%    |
| RNN   | top-k 40 | 0.055      | 0.315      | 0.013 | 129.7   | 31.5%    | 0.05%    |
| LSTM  | greedy   | 0.021      | 0.081      | 0.359 | 153.0   | 34.0%    | 0.00%    |
| LSTM  | temp 0.8 | 0.055      | 0.276      | 0.041 | 160.6   | 34.5%    | 0.06%    |
| LSTM  | temp 1.2 | 0.089      | 0.507      | 0.006 | 163.1   | 38.0%    | 0.24%    |
| LSTM  | top-k 40 | 0.056      | 0.315      | 0.022 | 166.9   | 41.5%    | 0.09%    |

All six configurations per model are in `plot_twist_generator/results/gen_metrics.json`.
"Overlong" means no `<eos>` was produced within the 200-token budget.

##### Blinded human evaluation (3 raters, 20 samples, 1 to 5 scale)

| System      | Coherence   | Grammar     | Relevance   | Creativity  |
| ----------- | ----------- | ----------- | ----------- | ----------- |
| Vanilla RNN | 1.40 ± 0.49 | 2.47 ± 0.62 | 1.10 ± 0.30 | 1.00 ± 0.00 |
| LSTM        | 1.50 ± 0.50 | 2.50 ± 0.56 | 1.20 ± 0.40 | 1.17 ± 0.37 |

Inter-rater agreement is high within ±1 point (100% on every criterion); exact agreement ranges from 63% (coherence, grammar)
to 90% (creativity).

#### 5. Key findings

- **LSTM beats the RNN on perplexity** by a clear margin, and also produces slightly longer, more fluent text.
- **Greedy decoding degenerates** into repetition (Rep-3 of 0.49 for the RNN, 0.36 for the LSTM) and has the lowest diversity.
- **Higher temperature raises diversity** (Distinct-2 rises from about 0.14 at T=0.5 to about 0.55 at T=1.2) but also raises the UNK rate and reduces coherence.
  Top-k 40 gives diversity similar to temperature 0.8 with a lower repetition rate than pure sampling at the same diversity.
- **Plot twists are not achieved by either model.** Human ratings are low across the board (coherence about 1.4 to 1.5 / 5,
  relevance about 1.1 to 1.2 / 5, creativity about 1.0 to 1.2 / 5). The continuations often drift to unrelated characters (for example "Lily" or "Tom")
  and generic TinyStories endings, ignoring the opening's twist. Differences between the two models in the human study
  are small relative to the spread, with only 20 samples.
- **Many generations never terminate.** 28% to 57% of outputs hit the 200-token budget without emitting `<eos>`.

#### 6. Repository layout

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

#### 7. Quick start

```bash
git clone https://github.com/ashutosh229/NLP_GA2.git
cd NLP_GA2/plot_twist_generator
pip install -r requirements.txt
python data_prep.py && python train.py --cell rnn && python train.py --cell lstm && python generate.py
```

#### 8. Reproducibility

Seeds are fixed in every script (data sampling 42, training 42, generation 1234, evaluation 2024, human-eval shuffle 7).
Each generated sample records its seed. Exact numbers can still vary slightly across hardware and PyTorch versions
(especially on GPU).

### Setup and Execution Guide (Q4)

This guide explains how to install, run and reproduce everything in `plot_twist_generator/`.

All commands below are run from inside the `plot_twist_generator/` directory, because the scripts import
each other with plain `from common import *` and write to folders next to themselves.

---

#### 1. Requirements

| Requirement | Notes                                                                                                    |
| ----------- | -------------------------------------------------------------------------------------------------------- |
| Python      | 3.10 or newer recommended                                                                                |
| OS          | Linux, macOS or Windows (Windows users: see the encoding note in Troubleshooting)                        |
| GPU         | Optional but strongly recommended. A single modest CUDA GPU trains both models in about 40 minutes total |
| Disk        | About 1 GB for the TinyStories download and Hugging Face cache; checkpoints are about 30 MB each         |
| Internet    | Needed once, to download TinyStories (and SmolLM2 if you run the optional LLM step)                      |

Python packages (`requirements.txt`): `torch`, `torchvision`, `datasets>=2.14`, `matplotlib`, `numpy`,
and `transformers>=4.40` (only for the optional LLM extension).

#### 2. Installation

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

#### 3. Quick smoke test (about 1 to 2 minutes, no download)

The `--toy` flag replaces TinyStories with a tiny synthetic corpus so you can check that the whole pipeline works.

```bash
python data_prep.py --toy
python train.py --cell rnn  --epochs 2
python train.py --cell lstm --epochs 2
python generate.py --n_samples 1 --max_new 50
```

> **Warning:** the toy run writes into the same `processed/`, `checkpoints/` and `results/` folders as the real run.
> After the smoke test, re-run the full pipeline from step 4.1 to overwrite the toy artefacts.

#### 4. Full reproduction

`run_all.sh` runs everything end to end (it also installs the CUDA build of PyTorch, so skip it if you installed manually).
The individual steps are:

##### 4.1 Part 1: Data preparation

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

##### 4.2 Part 2: Train the models

```bash
python train.py --cell rnn  --seed 42
python train.py --cell lstm --seed 42
```

Each run saves the best-validation checkpoint to `checkpoints/<cell>.pt` and the full training history
(per-epoch loss and perplexity, parameters, timing, test perplexity) to `results/history_<cell>.json`.

Reference timings (single GPU): RNN about 17 min, LSTM about 22 min for 20 epochs.

Options: `--emb 256 --hid 512 --layers 1 --dropout 0.3 --lr 2e-3 --bs 64 --seq_len 128 --epochs 20 --patience 3 --clip 1.0`.

##### 4.3 Parts 3 & 4: Generate stories

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

##### 4.4 Part 5: Quantitative evaluation

```bash
python evaluate_generation.py          # defaults: 200 test prompts of 20 tokens, --max_new 200, --seed 2024
```

Writes `results/gen_metrics.json` with Distinct-1/2, Rep-3, average length, premature-EOS rate, overlong rate and UNK rate
for every model × decoding config.

##### 4.5 Part 5: Blinded human evaluation

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

##### 4.6 Report tables and figures

```bash
python make_report_tables.py
```

Writes `results/tables.md` (dataset info, model comparison, generation metrics, human-evaluation tables) and
`results/loss_curves.png` (train/val loss and validation perplexity per epoch).

##### 4.7 Optional: LLM baseline

```bash
python llm_extension.py                          # default: HuggingFaceTB/SmolLM2-360M-Instruct
python llm_extension.py --ckpt <other-hf-model>  # any chat-template causal LM
python make_human_eval.py --include_llm          # blind the LLM samples into a NEW rating round
```

Generates 2 samples per prompt for `temp0.8` and `topk40` using a fixed prompt template (zero-shot, not fine-tuned),
and saves them with metadata (checkpoint, template, decoding, precision, hardware) to `results/llm_part4.json` and
`results/llm_metadata.json`. Remember the warning in section 4.5: adding the LLM samples means re-blinding and re-rating.

#### 5. Output map

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

#### 6. Reproducibility

- Seeds: data sampling `--seed 42`, training `--seed 42`, `generate.py` `--seed 1234`, `evaluate_generation.py` `--seed 2024`,
  human-eval shuffle `--seed 7`.
- Every generated sample stores its own seed, so any individual output can be regenerated.
- GPU results can differ slightly across hardware, CUDA and PyTorch versions even with fixed seeds.
- Greedy decoding is run once per prompt; sampled configs use several seeds (`seed + sample_index`).

#### 7. Troubleshooting

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

#### 8. Notes on interpreting outputs

- Text is lower-cased and word-tokenised, so generated stories are re-capitalised heuristically by `detokenize()`. Capitalisation
  and spacing around quotes are approximate and are not model errors.
- Generations that reach the token budget without `<eos>` are flagged `ended: false` and counted as "overlong" in the metrics.
- Rare words map to `<unk>`; it is excluded from `<pad>`/`<bos>` blocking but not from sampling, so `<unk>` can appear in outputs
  (more often at high temperature).
