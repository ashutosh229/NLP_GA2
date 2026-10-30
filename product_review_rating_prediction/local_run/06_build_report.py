"""Build the final <=6 page PDF report from results/*.json and results/*.png.
Run after 04_analyze.py and 05_per_class_report.py (results must already be
copied into ../results/, which is where this script reads from and writes to).
"""
import json
import os
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib import colors
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                                 Image as RLImage, KeepTogether)
from PIL import Image as PILImage

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.normpath(os.path.join(HERE, '..', 'results'))

all_res   = json.load(open(f'{RESULTS}/all_results.json'))
per_class = json.load(open(f'{RESULTS}/per_class_classification_report.json'))

styles = getSampleStyleSheet()
styles.add(ParagraphStyle('Small', parent=styles['Normal'], fontSize=8.5, leading=10.5))
styles.add(ParagraphStyle('H1', parent=styles['Heading1'], fontSize=13, spaceAfter=4, spaceBefore=6))
styles.add(ParagraphStyle('H2', parent=styles['Heading2'], fontSize=10.5, spaceAfter=3, spaceBefore=4))
styles.add(ParagraphStyle('BodySmall', parent=styles['Normal'], fontSize=9, leading=11.5))
styles.add(ParagraphStyle('Caption', parent=styles['Normal'], fontSize=7.5, leading=9,
                          textColor=colors.grey, alignment=TA_CENTER))
styles.add(ParagraphStyle('TitleCenter', parent=styles['Title'], fontSize=15, spaceAfter=2))
styles.add(ParagraphStyle('Subtitle', parent=styles['Normal'], fontSize=9.5, alignment=TA_CENTER,
                          textColor=colors.grey, spaceAfter=10))

def img_flowable(path, max_w_in):
    w, h = PILImage.open(path).size
    aspect = h / w
    return RLImage(path, width=max_w_in * inch, height=max_w_in * aspect * inch)

def table(data, col_widths=None, font_size=8):
    t = Table(data, colWidths=col_widths)
    t.setStyle(TableStyle([
        ('FONTSIZE', (0, 0), (-1, -1), font_size),
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c3e50')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f2f2f2')]),
        ('TOPPADDING', (0, 0), (-1, -1), 2),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
    ]))
    return t

story = []

# ── Title ────────────────────────────────────────────────────────────────────
story.append(Paragraph('Fine-Tuning RoBERTa for Multi-Class Product Review Rating Prediction', styles['TitleCenter']))
story.append(Paragraph('BiGRU + Additive Attention baseline vs. fine-tuned RoBERTa-base &#8212; Amazon Reviews (SetFit/amazon_reviews_multi_en)', styles['Subtitle']))

# ── 1. Objective & Dataset ──────────────────────────────────────────────────
story.append(Paragraph('1. Objective &amp; Dataset', styles['H1']))
story.append(Paragraph(
    'We implement and compare two 5-class (1-5 star) review-rating classifiers: a Bidirectional GRU '
    'with additive attention over frozen-then-fine-tuned fastText embeddings, and a fine-tuned '
    'RoBERTa-base model with masked mean pooling. Both are trained end-to-end on the full '
    'SetFit/amazon_reviews_multi_en dataset (200,000 train / 5,000 validation / 5,000 test reviews, '
    'title+body concatenated as input), with the test set evaluated only once, at the end.',
    styles['BodySmall']))

# ── 2. Architectures ─────────────────────────────────────────────────────────
story.append(Paragraph('2. Model Architectures', styles['H1']))
story.append(Paragraph('<b>Baseline (BiGRU):</b> Embedding (300-d fastText crawl-300d-2M, 98.2% vocabulary '
    'coverage; frozen epoch 1, fine-tuned at 0.1&#215; LR from epoch 2) &#8594; 2-layer Bidirectional GRU '
    '(hidden=256, dropout=0.3) &#8594; Bahdanau-style additive attention pooling &#8594; Dropout(0.3) &#8594; '
    'Linear(512&#8594;5). Input: spaCy-tokenized, lower-cased, HTML/URL-stripped text, vocab min-freq=3 '
    '(24,570 tokens), max length 200.', styles['BodySmall']))
story.append(Spacer(1, 3))
story.append(Paragraph('<b>RoBERTa:</b> roberta-base (125M params, byte-level BPE tokenizer, dynamic padding '
    'via a HF data collator) &#8594; masked mean pooling over all non-padded last-hidden-state tokens '
    '(task-required; not CLS pooling) &#8594; Dropout(0.1) &#8594; Linear(768&#8594;5), fine-tuned end-to-end. '
    'Main configuration: max length 256 tokens.', styles['BodySmall']))

# ── 3. Training Process ─────────────────────────────────────────────────────
story.append(Paragraph('3. Training Process', styles['H1']))
story.append(Paragraph(
    '<b>BiGRU:</b> Adam, LR=1e-3 (embeddings 1e-4 after unfreezing), batch=128, gradient clipping '
    '(max-norm 1.0), ReduceLROnPlateau on validation Macro-F1, early stopping (patience=3, max 15 '
    'epochs). <b>RoBERTa:</b> AdamW, LR=2e-5, weight decay 0.01 (no decay on bias/LayerNorm), batch=64, '
    'linear warmup (6%) + linear decay, fp16 mixed precision, early stopping (patience=2, max 3 epochs). '
    'Both: Cross-Entropy loss, fixed seeds, results reported as mean&#177;std over 2 seeds (42, 123). '
    'Training ran on a single GPU (Quadro RTX 8000, 48GB) using the full 200K-review training set for '
    'every run (no sub-sampling).', styles['BodySmall']))
story.append(Spacer(1, 3))
bigru42 = json.load(open(f'{RESULTS}/bigru_metrics_seed42.json'))
bigru123 = json.load(open(f'{RESULTS}/bigru_metrics_seed123.json'))
rb42 = json.load(open(f'{RESULTS}/roberta_metrics_main_s42.json'))
rb123 = json.load(open(f'{RESULTS}/roberta_metrics_main_s123.json'))
story.append(Paragraph(
    f'BiGRU converged (early-stopped) at epoch {len(bigru42["history"]["train_loss"])} (seed 42) / '
    f'{len(bigru123["history"]["train_loss"])} (seed 123); RoBERTa main used all 3 epochs for both seeds '
    '(monotonically improving validation Macro-F1 within budget). Training/validation loss and Macro-F1 '
    'curves (best seed per model) are shown below.', styles['BodySmall']))
story.append(Spacer(1, 4))

bg_curve_img = img_flowable(f'{RESULTS}/bigru_training_curves.png', 3.1)
rb_curve_img = img_flowable(f'{RESULTS}/roberta_training_curves.png', 3.1)
curve_table = Table([[bg_curve_img, rb_curve_img]], colWidths=[3.2 * inch, 3.2 * inch])
curve_table.setStyle(TableStyle([('ALIGN', (0, 0), (-1, -1), 'CENTER'), ('VALIGN', (0, 0), (-1, -1), 'TOP')]))
story.append(curve_table)
story.append(Paragraph('Figure 1: Training/validation loss and Macro-F1/Accuracy, BiGRU (left, seed 42) vs. RoBERTa main (right, seed 42).', styles['Caption']))

# ── 4. Quantitative Results ─────────────────────────────────────────────────
story.append(Paragraph('4. Quantitative Results (Test Set)', styles['H1']))
bg = all_res['bigru']; rb = all_res['roberta_main']; ab = all_res['ablation_b_mean128']
qdata = [
    ['Model', 'Accuracy', 'Macro F1', 'MAE'],
    ['BiGRU + Attention (baseline)', f'{bg["acc_mean"]:.3f} ± {bg["acc_std"]:.3f}',
     f'{bg["mf1_mean"]:.3f} ± {bg["mf1_std"]:.3f}', f'{bg["mae_mean"]:.3f} ± {bg["mae_std"]:.3f}'],
    ['RoBERTa, mean-pool, len=256 (main)', f'{rb["acc_mean"]:.3f} ± {rb["acc_std"]:.3f}',
     f'{rb["mf1_mean"]:.3f} ± {rb["mf1_std"]:.3f}', f'{rb["mae_mean"]:.3f} ± {rb["mae_std"]:.3f}'],
    ['RoBERTa, mean-pool, len=128 (Ablation B)', f'{ab["acc"]:.3f}', f'{ab["mf1"]:.3f}', f'{ab["mae"]:.3f}'],
]
story.append(table(qdata, col_widths=[2.5 * inch, 1.4 * inch, 1.4 * inch, 1.1 * inch]))
story.append(Spacer(1, 5))

bg_pc = per_class['bigru']['report']; rb_pc = per_class['roberta']['report']
star_names = [f'{i+1}-star' for i in range(5)]
pc_data = [['Class', 'BiGRU F1', 'RoBERTa F1']] + [
    [s, f'{bg_pc[s]["f1-score"]:.3f}', f'{rb_pc[s]["f1-score"]:.3f}'] for s in star_names
]
pc_table = table(pc_data, col_widths=[1.1 * inch, 1.1 * inch, 1.1 * inch], font_size=8)

bg_ord = bg['ordinal_error']; rb_ord = rb['ordinal_error']
ord_data = [['Error distance', 'BiGRU', 'RoBERTa'],
            ['Correct (0)', f'{bg_ord["correct_diff0"]:.3f}', f'{rb_ord["correct_diff0"]:.3f}'],
            ['Adjacent (1)', f'{bg_ord["adjacent_diff1"]:.3f}', f'{rb_ord["adjacent_diff1"]:.3f}'],
            ['Near-miss (2)', f'{bg_ord["near_miss_diff2"]:.3f}', f'{rb_ord["near_miss_diff2"]:.3f}'],
            ['Distant (>=3)', f'{bg_ord["distant_diff_ge3"]:.3f}', f'{rb_ord["distant_diff_ge3"]:.3f}']]
ord_table = table(ord_data, col_widths=[1.3 * inch, 1.0 * inch, 1.0 * inch], font_size=8)

side_by_side = Table([[pc_table, Spacer(0.3 * inch, 1), ord_table]])
side_by_side.setStyle(TableStyle([('VALIGN', (0, 0), (-1, -1), 'TOP')]))
story.append(side_by_side)
story.append(Paragraph('Table: per-class F1 (left) and ordinal error distribution |pred-true| (right, fraction of test set).', styles['Caption']))
story.append(Spacer(1, 5))

cm_img = img_flowable(f'{RESULTS}/confusion_matrices.png', 6.4)
story.append(cm_img)
story.append(Paragraph('Figure 2: Row-normalized confusion matrices on the test set.', styles['Caption']))

# ── 5. Ablation Study ───────────────────────────────────────────────────────
story.append(Paragraph('5. Ablation Study: Max Sequence Length (256 vs. 128 tokens)', styles['H1']))
acc_drop_pts = (rb42['acc'] - ab['acc']) * 100
mf1_drop_pts = (rb42['mf1'] - ab['mf1']) * 100
story.append(Paragraph(
    'Chosen ablation: mean pooling at max_len=128 ("Ablation B") vs. the main configuration '
    '(mean pooling, max_len=256), both seed 42. Truncating to 128 tokens costs only '
    f'{acc_drop_pts:.2f} accuracy points ({rb42["acc"]*100:.2f}%→{ab["acc"]*100:.2f}%) and '
    f'{mf1_drop_pts:.2f} Macro-F1 points ({rb42["mf1"]*100:.2f}%→{ab["mf1"]*100:.2f}%), with MAE '
    f'essentially unchanged ({rb42["mae"]:.3f}→{ab["mae"]:.3f}). This indicates most of the '
    'rating-relevant sentiment signal in these reviews is concentrated in the first ~128 BPE tokens '
    '(title + opening sentences); halving the sequence length roughly halves per-epoch compute '
    '(measured ~420s vs. ~630s/epoch on the same GPU) for a near-negligible accuracy cost &#8212; a '
    'favorable efficiency/accuracy trade-off if compute is constrained.', styles['BodySmall']))

# ── 6. Qualitative Analysis ─────────────────────────────────────────────────
story.append(Paragraph('6. Qualitative Analysis', styles['H1']))
story.append(Paragraph(
    f'{all_res["bigru_fails_roberta_wins"]} test reviews are mispredicted by BiGRU but correctly rated by '
    f'RoBERTa, vs. {all_res["roberta_fails_bigru_wins"]} the other way. Inspecting examples of each '
    '(full text in results/qualitative_examples.txt, full analysis in results/qualitative_analysis.md):',
    styles['BodySmall']))
story.append(Spacer(1, 2))
story.append(Paragraph(
    '<b>RoBERTa wins on:</b> contrastive-conjunction reviews ("good, <i>but</i>...", "very happy, '
    '<i>however</i>...") where an early strong-polarity clause is overturned later in the text; figurative '
    'language/idioms ("ripping my hair out") that are lexically strong-negative but contextually mild; '
    'sarcasm ("thank goodness it was a free book") requiring magnitude calibration, not just polarity; and '
    'long, informal reviews with slang/interjections where BiGRU\'s fixed 200-token window and fastText '
    'OOV handling lose resolving context. RoBERTa\'s full self-attention over BPE subwords integrates '
    'this long-range, contrastive structure far more reliably than BiGRU\'s recurrent+attention pooling.',
    styles['BodySmall']))
story.append(Spacer(1, 2))
story.append(Paragraph(
    '<b>BiGRU wins on:</b> very short, uniformly positive reviews with no contrast word, where the true '
    'label (4 vs. 5 stars) is arguably under-determined by the text and RoBERTa\'s stronger positive-'
    'sentiment prior overshoots to 5; and mild comparative-negation phrasing ("doesn\'t work any better '
    'than...") that is lexically negative but semantically near-neutral, where RoBERTa over-weights the '
    'negation cue.', styles['BodySmall']))
story.append(Spacer(1, 4))
ti_img = img_flowable(f'{RESULTS}/token_importance.png', 6.4)
story.append(ti_img)
story.append(Paragraph(
    'Figure 3: Gradient-norm token-importance (saliency w.r.t. input embeddings, true-class logit) for a '
    'true=1-star review ("...I’m very disappointed in the service", wrong case size shipped). Importance '
    'concentrates on "disappointed" and the negation/mismatch tokens late in the review rather than the '
    'mildly positive opener ("looks cool") &#8212; consistent with RoBERTa correctly identifying the decisive '
    'complaint rather than anchoring on the early token, the same behavior underlying its wins above.',
    styles['Caption']))

# ── 7. Comparative Analysis & Conclusion ────────────────────────────────────
story.append(Paragraph('7. Comparative Analysis &amp; Conclusion', styles['H1']))
story.append(Paragraph(
    f'RoBERTa outperforms the BiGRU baseline on every metric: +{(rb["acc_mean"]-bg["acc_mean"])*100:.1f} pts '
    f'accuracy, +{(rb["mf1_mean"]-bg["mf1_mean"])*100:.1f} pts Macro-F1, and {bg["mae_mean"]-rb["mae_mean"]:.3f} '
    'lower MAE, with lower variance across seeds. Both models respect the ordinal nature of the task '
    f'reasonably well &#8212; errors are overwhelmingly adjacent-class (BiGRU {bg_ord["adjacent_diff1"]*100:.1f}%, '
    f'RoBERTa {rb_ord["adjacent_diff1"]*100:.1f}%) rather than distant (BiGRU {bg_ord["distant_diff_ge3"]*100:.1f}%, '
    f'RoBERTa {rb_ord["distant_diff_ge3"]*100:.1f}%) &#8212; but RoBERTa is both more accurate and slightly better '
    'calibrated ordinally (higher exact-match rate, lower near-miss/distant rate). Per-class F1 shows '
    'both models struggle most on the middle classes (2- and 3-star), which are intrinsically more '
    'ambiguous (mixed sentiment is common there), and are strongest on the extremes (1- and 5-star), '
    'where sentiment polarity is least ambiguous. The qualitative analysis explains the quantitative gap: '
    'RoBERTa\'s contextual, long-range self-attention over subwords handles contrastive conjunctions, '
    'idioms, sarcasm, and informal/long text substantially better than BiGRU\'s fixed-window recurrent '
    'pooling over static-then-fine-tuned word embeddings, at the cost of far more compute. The length '
    'ablation shows that cost can be partly reclaimed (half the sequence length, negligible accuracy '
    'loss) without sacrificing RoBERTa\'s architectural advantage.', styles['BodySmall']))

doc = SimpleDocTemplate(f'{RESULTS}/report.pdf', pagesize=letter,
                        leftMargin=0.6 * inch, rightMargin=0.6 * inch,
                        topMargin=0.5 * inch, bottomMargin=0.5 * inch)
doc.build(story)

from pypdf import PdfReader
n = len(PdfReader(f'{RESULTS}/report.pdf').pages)
print(f'Built {RESULTS}/report.pdf : {n} pages')
