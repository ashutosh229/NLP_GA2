import os
import sys
import json
import pickle
import math
import torch
import sacrebleu
from typing import List, Dict, Tuple
from tqdm import tqdm

# Ensure UTF-8 output
sys.stdout.reconfigure(encoding="utf-8")

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from data_preprocessing import (
    Vocabulary, PAD_IDX, SOS_IDX, EOS_IDX, UNK_IDX,
    normalize_english, normalize_hindi, tokenize_english, tokenize_hindi
)
sys.modules['__main__'].Vocabulary = Vocabulary
from models import (
    BaselineEncoder, BaselineDecoder, Seq2SeqBaseline,
    AttentionEncoder, AttentionDecoder, Seq2SeqAttention
)

def load_trained_models(device: str = "cpu", embed_dim: int = 128, hidden_dim: int = 256):
    data_path = os.path.join(current_dir, "data", "preprocessed_25000_30.pkl")
    with open(data_path, "rb") as f:
        data = pickle.load(f)

    src_vocab = data["src_vocab"]
    tgt_vocab = data["tgt_vocab"]
    test_pairs = data["test_pairs"]

    base_enc = BaselineEncoder(len(src_vocab), embed_dim, hidden_dim, num_layers=1, dropout=0.2)
    base_dec = BaselineDecoder(len(tgt_vocab), embed_dim, hidden_dim, num_layers=1, dropout=0.2)
    baseline_model = Seq2SeqBaseline(base_enc, base_dec, pad_idx=PAD_IDX).to(device)

    attn_enc = AttentionEncoder(len(src_vocab), embed_dim, hidden_dim, num_layers=1, dropout=0.2)
    attn_dec = AttentionDecoder(len(tgt_vocab), embed_dim, hidden_dim, num_layers=1, dropout=0.2)
    attention_model = Seq2SeqAttention(attn_enc, attn_dec, pad_idx=PAD_IDX).to(device)

    checkpoint_dir = os.path.join(current_dir, "checkpoints")
    base_ckpt = os.path.join(checkpoint_dir, "baseline_lstm_best.pt")
    attn_ckpt = os.path.join(checkpoint_dir, "attention_lstm_best.pt")

    if os.path.exists(base_ckpt):
        baseline_model.load_state_dict(torch.load(base_ckpt, map_location=device))
        print(f"Loaded baseline checkpoint from {base_ckpt}")
    else:
        print(f"Warning: Baseline checkpoint not found at {base_ckpt}")

    if os.path.exists(attn_ckpt):
        attention_model.load_state_dict(torch.load(attn_ckpt, map_location=device))
        print(f"Loaded attention checkpoint from {attn_ckpt}")
    else:
        print(f"Warning: Attention checkpoint not found at {attn_ckpt}")

    return baseline_model, attention_model, src_vocab, tgt_vocab, test_pairs

def translate_sentence_baseline(model, en_tokens: List[str], src_vocab, tgt_vocab, device: str = "cpu", max_len: int = 50):
    model.eval()
    src_ids = src_vocab.encode(en_tokens, add_sos_eos=True)
    src_tensor = torch.tensor(src_ids, dtype=torch.long, device=device).unsqueeze(0)
    decoded_ids = model.translate_sentence(src_tensor, max_len=max_len, sos_idx=SOS_IDX, eos_idx=EOS_IDX)
    decoded_tokens = tgt_vocab.decode(decoded_ids, remove_special=True)
    return decoded_tokens

def translate_sentence_attention(model, en_tokens: List[str], src_vocab, tgt_vocab, device: str = "cpu", max_len: int = 50):
    model.eval()
    src_ids = src_vocab.encode(en_tokens, add_sos_eos=True)
    src_tensor = torch.tensor(src_ids, dtype=torch.long, device=device).unsqueeze(0)
    decoded_ids, attention_matrix = model.translate_sentence(src_tensor, max_len=max_len, sos_idx=SOS_IDX, eos_idx=EOS_IDX)
    decoded_tokens = tgt_vocab.decode(decoded_ids, remove_special=True)
    return decoded_tokens, attention_matrix

def compute_corpus_bleu(hypotheses: List[str], references: List[str]) -> float:
    if not hypotheses or not references:
        return 0.0
    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    return bleu.score

def evaluate_models_on_test(
    baseline_model,
    attention_model,
    src_vocab,
    tgt_vocab,
    test_pairs,
    device: str = "cpu",
    max_eval_samples: int = 1500
):
    print(f"\nEvaluating models on {min(len(test_pairs), max_eval_samples)} test sentences...")

    # Length buckets: 1-5, 6-10, 11-15, 16+ tokens
    buckets = {
        "1-5 tokens": [],
        "6-10 tokens": [],
        "11-15 tokens": [],
        "16+ tokens": []
    }

    eval_pairs = test_pairs[:max_eval_samples]

    for en_tokens, hi_tokens in eval_pairs:
        L = len(en_tokens)
        if 1 <= L <= 5:
            buckets["1-5 tokens"].append((en_tokens, hi_tokens))
        elif 6 <= L <= 10:
            buckets["6-10 tokens"].append((en_tokens, hi_tokens))
        elif 11 <= L <= 15:
            buckets["11-15 tokens"].append((en_tokens, hi_tokens))
        else:
            buckets["16+ tokens"].append((en_tokens, hi_tokens))

    results_by_bucket = {}
    all_refs = []
    all_base_hyps = []
    all_attn_hyps = []

    for bucket_name, pairs in buckets.items():
        if not pairs:
            continue
        print(f"Evaluating bucket: {bucket_name} ({len(pairs)} sentences)...")
        b_refs = []
        b_base_hyps = []
        b_attn_hyps = []

        for en_toks, hi_toks in pairs:
            ref_str = " ".join(hi_toks)
            base_pred_toks = translate_sentence_baseline(baseline_model, en_toks, src_vocab, tgt_vocab, device=device)
            attn_pred_toks, _ = translate_sentence_attention(attention_model, en_toks, src_vocab, tgt_vocab, device=device)

            base_str = " ".join(base_pred_toks) if base_pred_toks else "<EMPTY>"
            attn_str = " ".join(attn_pred_toks) if attn_pred_toks else "<EMPTY>"

            b_refs.append(ref_str)
            b_base_hyps.append(base_str)
            b_attn_hyps.append(attn_str)

        all_refs.extend(b_refs)
        all_base_hyps.extend(b_base_hyps)
        all_attn_hyps.extend(b_attn_hyps)

        base_bleu = compute_corpus_bleu(b_base_hyps, b_refs)
        attn_bleu = compute_corpus_bleu(b_attn_hyps, b_refs)

        results_by_bucket[bucket_name] = {
            "num_sentences": len(pairs),
            "baseline_bleu": round(base_bleu, 2),
            "attention_bleu": round(attn_bleu, 2),
            "relative_improvement_pct": round(((attn_bleu - base_bleu) / max(base_bleu, 1e-5)) * 100, 2)
        }

    overall_base_bleu = compute_corpus_bleu(all_base_hyps, all_refs)
    overall_attn_bleu = compute_corpus_bleu(all_attn_hyps, all_refs)

    summary = {
        "overall": {
            "total_evaluated": len(eval_pairs),
            "baseline_bleu": round(overall_base_bleu, 2),
            "attention_bleu": round(overall_attn_bleu, 2),
            "improvement": round(overall_attn_bleu - overall_base_bleu, 2)
        },
        "by_sentence_length": results_by_bucket
    }

    print("\n================== EVALUATION RESULTS ==================")
    print(f"Overall Baseline BLEU:  {overall_base_bleu:.2f}")
    print(f"Overall Attention BLEU: {overall_attn_bleu:.2f} (+{overall_attn_bleu - overall_base_bleu:.2f})")
    print("-------------------------------------------------------")
    print(f"{'Length Bucket':<15} | {'Count':<7} | {'Baseline':<10} | {'Attention':<10} | {'Diff'}")
    print("-------------------------------------------------------")
    for b_name, b_res in results_by_bucket.items():
        diff = b_res['attention_bleu'] - b_res['baseline_bleu']
        print(f"{b_name:<15} | {b_res['num_sentences']:<7} | {b_res['baseline_bleu']:<10.2f} | {b_res['attention_bleu']:<10.2f} | {diff:+.2f}")
    print("=======================================================\n")

    os.makedirs(os.path.join(current_dir, "data"), exist_ok=True)
    out_file = os.path.join(current_dir, "data", "evaluation_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)
    print(f"Evaluation results saved to {out_file}")

    return summary

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_m, attn_m, s_voc, t_voc, t_pairs = load_trained_models(device=device)
    evaluate_models_on_test(base_m, attn_m, s_voc, t_voc, t_pairs, device=device, max_eval_samples=1000)
