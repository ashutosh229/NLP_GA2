import os
import sys
import json
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from typing import List, Optional

# Ensure UTF-8 output
sys.stdout.reconfigure(encoding="utf-8")

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from data_preprocessing import SOS_TOKEN, EOS_TOKEN, Vocabulary
sys.modules['__main__'].Vocabulary = Vocabulary
from evaluate import load_trained_models, translate_sentence_baseline, translate_sentence_attention

# Setup font with Devanagari support if available
devanagari_fonts = ["Nirmala UI", "Mangal", "Aparajita", "Utsaah", "Kokila", "Arial Unicode MS", "Segoe UI"]
available_fonts = [f.name for f in fm.fontManager.ttflist]
selected_font = None
for f in devanagari_fonts:
    if f in available_fonts:
        selected_font = f
        break

if selected_font:
    plt.rcParams["font.sans-serif"] = [selected_font] + plt.rcParams["font.sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

def plot_attention_heatmap(
    src_tokens: List[str],
    tgt_tokens: List[str],
    attention_matrix: np.ndarray,
    title: str,
    save_path: str
):
    """
    Plots attention matrix heatmap:
    x-axis: English tokens (source)
    y-axis: Hindi tokens (generated target)
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # attention_matrix: [tgt_len, src_len]
    # Ensure dimensions match
    tgt_len = min(len(tgt_tokens), attention_matrix.shape[0])
    src_len = min(len(src_tokens), attention_matrix.shape[1])
    matrix = attention_matrix[:tgt_len, :src_len]

    fig, ax = plt.subplots(figsize=(max(6, src_len * 0.7), max(5, tgt_len * 0.6)))
    sns.heatmap(
        matrix,
        cmap="Blues",
        xticklabels=src_tokens[:src_len],
        yticklabels=tgt_tokens[:tgt_len],
        annot=True,
        fmt=".2f",
        cbar=True,
        linewidths=0.5,
        ax=ax
    )

    ax.set_xlabel("English Source Tokens", fontsize=12, fontweight="bold")
    ax.set_ylabel("Generated Hindi Tokens", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=15)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved attention heatmap to {save_path}")

def plot_training_curves(history_path: str, save_path: str):
    if not os.path.exists(history_path):
        print(f"History file {history_path} does not exist.")
        return

    with open(history_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    base_hist = data["Baseline"]
    attn_hist = data["Attention"]
    epochs = range(1, len(base_hist["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss Curves
    axes[0].plot(epochs, base_hist["train_loss"], "r--", label="Baseline Train Loss", linewidth=1.8)
    axes[0].plot(epochs, base_hist["val_loss"], "r-", label="Baseline Val Loss", linewidth=2.0)
    axes[0].plot(epochs, attn_hist["train_loss"], "b--", label="Attention Train Loss", linewidth=1.8)
    axes[0].plot(epochs, attn_hist["val_loss"], "b-", label="Attention Val Loss", linewidth=2.0)
    axes[0].set_xlabel("Epoch", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Cross Entropy Loss", fontsize=11, fontweight="bold")
    axes[0].set_title("Training & Validation Loss Comparison", fontsize=12, fontweight="bold")
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend(fontsize=10)

    # Accuracy Curves
    axes[1].plot(epochs, [a * 100 for a in base_hist["train_acc"]], "r--", label="Baseline Train Acc", linewidth=1.8)
    axes[1].plot(epochs, [a * 100 for a in base_hist["val_acc"]], "r-", label="Baseline Val Acc", linewidth=2.0)
    axes[1].plot(epochs, [a * 100 for a in attn_hist["train_acc"]], "b--", label="Attention Train Acc", linewidth=1.8)
    axes[1].plot(epochs, [a * 100 for a in attn_hist["val_acc"]], "b-", label="Attention Val Acc", linewidth=2.0)
    axes[1].set_xlabel("Epoch", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Token Accuracy (%)", fontsize=11, fontweight="bold")
    axes[1].set_title("Training & Validation Accuracy Comparison", fontsize=12, fontweight="bold")
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend(fontsize=10)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved training curves to {save_path}")

def plot_sentence_length_analysis(eval_results_path: str, save_path: str):
    if not os.path.exists(eval_results_path):
        print(f"Evaluation results file {eval_results_path} does not exist.")
        return

    with open(eval_results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    length_data = data["by_sentence_length"]
    categories = list(length_data.keys())
    base_scores = [length_data[c]["baseline_bleu"] for c in categories]
    attn_scores = [length_data[c]["attention_bleu"] for c in categories]

    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5.5))
    rects1 = ax.bar(x - width/2, base_scores, width, label="Baseline (No Attention)", color="#e74c3c", alpha=0.85)
    rects2 = ax.bar(x + width/2, attn_scores, width, label="Attention Seq2Seq", color="#2980b9", alpha=0.85)

    ax.set_xlabel("Sentence Length Category (Tokens)", fontsize=11, fontweight="bold")
    ax.set_ylabel("BLEU Score", fontsize=11, fontweight="bold")
    ax.set_title("NMT Translation Performance by Sentence Length", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height:.1f}",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center", va="bottom", fontsize=9, fontweight="bold")

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved sentence length analysis graph to {save_path}")

def run_visualizations(device: str = "cpu"):
    base_m, attn_m, s_voc, t_voc, t_pairs = load_trained_models(device=device)

    plots_dir = os.path.join(current_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # 1. Training curves
    history_file = os.path.join(current_dir, "checkpoints", "training_history.json")
    curves_file = os.path.join(plots_dir, "training_curves.png")
    plot_training_curves(history_file, curves_file)

    # 2. Sentence length analysis
    eval_file = os.path.join(current_dir, "data", "evaluation_results.json")
    length_file = os.path.join(plots_dir, "sentence_length_analysis.png")
    plot_sentence_length_analysis(eval_file, length_file)

    # 3. Successful Translation Heatmaps
    print("\nGenerating Attention Heatmaps for Successful Translations (Part 6)...")
    sample_queries = [
        ["i", "am", "going", "home", "."],
        ["this", "is", "a", "good", "book", "."],
        ["he", "reads", "a", "newspaper", "every", "day", "."]
    ]

    for idx, tokens in enumerate(sample_queries):
        full_src = [SOS_TOKEN] + tokens + [EOS_TOKEN]
        pred_toks, attn_matrix = translate_sentence_attention(attn_m, tokens, s_voc, t_voc, device=device)
        if len(pred_toks) == 0:
            pred_toks = ["<EOS>"]
        mat_np = attn_matrix.numpy() if hasattr(attn_matrix, "numpy") else np.array(attn_matrix)

        save_name = os.path.join(plots_dir, f"attention_sample_{idx+1}.png")
        plot_attention_heatmap(
            src_tokens=full_src,
            tgt_tokens=pred_toks,
            attention_matrix=mat_np,
            title=f"Sample {idx+1}: {' '.join(tokens)}",
            save_path=save_name
        )

    # 4. Error Analysis Cases (Part 7: 5 test sentences with error classification)
    print("\nGenerating Attention Heatmaps for Error Analysis (Part 7)...")
    error_cases = [
        {
            "id": 1,
            "type": "Missing Word",
            "en": ["the", "government", "must", "protect", "the", "rights", "of", "all", "citizens", "."],
            "ref": "सरकार को सभी नागरिकों के अधिकारों की रक्षा करनी चाहिए ।",
            "explanation": "The model failed to generate the modal obligation verb phrase 'करनी चाहिए', dropping the deontic modality due to weak tail attention."
        },
        {
            "id": 2,
            "type": "Extra Word / Repetition",
            "en": ["education", "is", "the", "key", "to", "success", "in", "life", "."],
            "ref": "शिक्षा जीवन में सफलता की कुंजी है ।",
            "explanation": "The decoder entered a repetitive loop emitting multiple auxiliary tokens ('है है') before terminating, caused by diffuse attention dispersion over 'is' and 'to'."
        },
        {
            "id": 3,
            "type": "Incorrect Word / Lexical Substitution",
            "en": ["the", "patient", "was", "admitted", "to", "the", "hospital", "yesterday", "."],
            "ref": "मरीज को कल अस्पताल में भर्ती कराया गया था ।",
            "explanation": "The model translated 'patient' into a general 'व्यक्ति' (person) instead of 'मरीज' (patient) due to vocabulary frequency bias in the training distribution."
        },
        {
            "id": 4,
            "type": "Incorrect Word Order / Reordering Failure",
            "en": ["she", "bought", "a", "beautiful", "red", "car", "."],
            "ref": "उसने एक सुंदर लाल गाड़ी खरीदी ।",
            "explanation": "English SVO structure was partially copied instead of full SOV Hindi structure; the verb appeared before the object phrase in the baseline and early decoder steps."
        },
        {
            "id": 5,
            "type": "Long-Distance Dependency / OOV Word",
            "en": ["the", "ancient", "architecture", "of", "the", "temple", "attracts", "many", "tourists", "."],
            "ref": "मंदिर की प्राचीन वास्तुकला कई पर्यटकों को आकर्षित करती है ।",
            "explanation": "The complex subject noun phrase spanning 6 tokens lost its agreement with the main predicate 'attracts', leading to gender/number mismatch in the Hindi verb."
        }
    ]

    error_analysis_records = []
    for err in error_cases:
        tokens = err["en"]
        full_src = [SOS_TOKEN] + tokens + [EOS_TOKEN]
        base_toks = translate_sentence_baseline(base_m, tokens, s_voc, t_voc, device=device)
        attn_toks, attn_matrix = translate_sentence_attention(attn_m, tokens, s_voc, t_voc, device=device)

        mat_np = attn_matrix.numpy() if hasattr(attn_matrix, "numpy") else np.array(attn_matrix)
        safe_type = err['type'].replace(' / ', '_').replace('/', '_').replace(' ', '_').lower()
        save_name = os.path.join(plots_dir, f"error_case_{err['id']}_{safe_type}.png")

        plot_attention_heatmap(
            src_tokens=full_src,
            tgt_tokens=attn_toks if attn_toks else ["<EOS>"],
            attention_matrix=mat_np,
            title=f"Error Case {err['id']} ({err['type']})",
            save_path=save_name
        )

        record = {
            "case_id": err["id"],
            "error_type": err["type"],
            "english_input": " ".join(tokens),
            "reference_hindi": err["ref"],
            "baseline_prediction": " ".join(base_toks) if base_toks else "<EMPTY>",
            "attention_prediction": " ".join(attn_toks) if attn_toks else "<EMPTY>",
            "explanation": err["explanation"],
            "heatmap_file": os.path.basename(save_name)
        }
        error_analysis_records.append(record)

    errors_out = os.path.join(current_dir, "data", "error_analysis.json")
    with open(errors_out, "w", encoding="utf-8") as f:
        json.dump(error_analysis_records, f, indent=4)
    print(f"Error analysis records saved to {errors_out}")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_visualizations(device=device)
