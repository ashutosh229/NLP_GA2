import os
import sys
import json
import time
import random
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Ensure UTF-8 output
sys.stdout.reconfigure(encoding="utf-8")

# Add current folder to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from data_preprocessing import (
    Vocabulary, TranslationDataset, collate_fn,
    PAD_IDX, SOS_IDX, EOS_IDX, UNK_IDX, prepare_data
)
from models import (
    BaselineEncoder, BaselineDecoder, Seq2SeqBaseline,
    AttentionEncoder, AttentionDecoder, Seq2SeqAttention
)

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(min(8, os.cpu_count() or 4))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def compute_accuracy(preds: torch.Tensor, targets: torch.Tensor, pad_idx: int = PAD_IDX) -> float:
    # preds: [batch_size, tgt_len, vocab_size]
    # targets: [batch_size, tgt_len]
    pred_tokens = preds.argmax(dim=-1)
    mask = (targets != pad_idx)
    correct = (pred_tokens == targets) & mask
    total = mask.sum().item()
    if total == 0:
        return 0.0
    return correct.sum().item() / total

def train_epoch(model, dataloader, optimizer, criterion, clip: float = 1.0, teacher_forcing_ratio: float = 0.5, device: str = "cpu"):
    model.train()
    total_loss = 0.0
    total_acc = 0.0

    for step, (src, src_lens, tgt, tgt_lens, _, _) in enumerate(dataloader, 1):
        src = src.to(device)
        tgt = tgt.to(device)

        optimizer.zero_grad()

        # Forward pass
        if isinstance(model, Seq2SeqAttention):
            output, _ = model(src, tgt, teacher_forcing_ratio=teacher_forcing_ratio)
        else:
            output = model(src, tgt, teacher_forcing_ratio=teacher_forcing_ratio)

        # Output: [batch_size, tgt_len, vocab_size]
        # Ignore first token (<SOS>) when computing loss, evaluate targets[:, 1:]
        output_dim = output.shape[-1]
        output_flattened = output[:, 1:].reshape(-1, output_dim)
        tgt_flattened = tgt[:, 1:].reshape(-1)

        loss = criterion(output_flattened, tgt_flattened)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        acc = compute_accuracy(output[:, 1:], tgt[:, 1:], pad_idx=PAD_IDX)

        total_loss += loss.item()
        total_acc += acc

        if step % 50 == 0 or step == len(dataloader):
            print(f"  Step {step}/{len(dataloader)} | Loss: {loss.item():.4f} | Acc: {acc*100:.2f}%", flush=True)

    num_batches = len(dataloader)
    return total_loss / num_batches, total_acc / num_batches

def evaluate_epoch(model, dataloader, criterion, device: str = "cpu"):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0

    with torch.no_grad():
        for src, src_lens, tgt, tgt_lens, _, _ in dataloader:
            src = src.to(device)
            tgt = tgt.to(device)

            if isinstance(model, Seq2SeqAttention):
                output, _ = model(src, tgt, teacher_forcing_ratio=1.0)  # Standard evaluation cross-entropy
            else:
                output = model(src, tgt, teacher_forcing_ratio=1.0)

            output_dim = output.shape[-1]
            output_flattened = output[:, 1:].reshape(-1, output_dim)
            tgt_flattened = tgt[:, 1:].reshape(-1)

            loss = criterion(output_flattened, tgt_flattened)
            acc = compute_accuracy(output[:, 1:], tgt[:, 1:], pad_idx=PAD_IDX)

            total_loss += loss.item()
            total_acc += acc

    num_batches = len(dataloader)
    return total_loss / num_batches, total_acc / num_batches

def train_system(
    model_name: str,
    model,
    train_loader,
    val_loader,
    epochs: int = 10,
    lr: float = 0.001,
    checkpoint_dir: str = "checkpoints",
    device: str = "cpu"
):
    print(f"\n==========================================")
    print(f"Starting Training: {model_name}")
    print(f"Device: {device} | Epochs: {epochs} | LR: {lr}")
    print(f"==========================================")

    os.makedirs(checkpoint_dir, exist_ok=True)
    best_model_path = os.path.join(checkpoint_dir, f"{model_name.lower().replace(' ', '_')}_best.pt")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "epoch_times": []
    }

    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, clip=1.0, teacher_forcing_ratio=0.5, device=device
        )
        val_loss, val_acc = evaluate_epoch(model, val_loader, criterion, device=device)
        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["epoch_times"].append(elapsed)

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            star = " [* Best Val Loss]"
        else:
            star = ""

        print(
            f"Epoch {epoch:02d}/{epochs:02d} [{elapsed:.1f}s] | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%{star}",
            flush=True
        )

    # Save final history
    return history, best_model_path

def main(epochs: int = 10, batch_size: int = 64, embed_dim: int = 128, hidden_dim: int = 256):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset
    data_path = os.path.join(current_dir, "data", "preprocessed_25000_30.pkl")
    if not os.path.exists(data_path):
        print(f"Preprocessed data not found at {data_path}. Running data preparation...")
        data = prepare_data(train_size=25000, max_len=30, min_freq=2, cache_dir=os.path.join(current_dir, "data"))
    else:
        print(f"Loading data from {data_path}...")
        with open(data_path, "rb") as f:
            data = pickle.load(f)

    train_pairs = data["train_pairs"]
    val_pairs = data["val_pairs"]
    test_pairs = data["test_pairs"]
    src_vocab = data["src_vocab"]
    tgt_vocab = data["tgt_vocab"]

    print(f"Train samples: {len(train_pairs)}, Val samples: {len(val_pairs)}, Test samples: {len(test_pairs)}")
    print(f"English vocab: {len(src_vocab)}, Hindi vocab: {len(tgt_vocab)}")

    train_dataset = TranslationDataset(train_pairs, src_vocab, tgt_vocab)
    val_dataset = TranslationDataset(val_pairs, src_vocab, tgt_vocab)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    checkpoint_dir = os.path.join(current_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 1. Build and Train Model A: Baseline without Attention
    print("\n--- Initializing Model A: Baseline LSTM Encoder-Decoder (NO Attention) ---")
    base_enc = BaselineEncoder(len(src_vocab), embed_dim, hidden_dim, num_layers=1, dropout=0.2)
    base_dec = BaselineDecoder(len(tgt_vocab), embed_dim, hidden_dim, num_layers=1, dropout=0.2)
    baseline_model = Seq2SeqBaseline(base_enc, base_dec, pad_idx=PAD_IDX).to(device)

    base_history, base_path = train_system(
        "Baseline_LSTM",
        baseline_model,
        train_loader,
        val_loader,
        epochs=epochs,
        lr=0.001,
        checkpoint_dir=checkpoint_dir,
        device=device
    )

    # 2. Build and Train Model B: Attention-Based Seq2Seq
    print("\n--- Initializing Model B: Attention-Based LSTM Encoder-Decoder (WITH Bahdanau Attention) ---")
    attn_enc = AttentionEncoder(len(src_vocab), embed_dim, hidden_dim, num_layers=1, dropout=0.2)
    attn_dec = AttentionDecoder(len(tgt_vocab), embed_dim, hidden_dim, num_layers=1, dropout=0.2)
    attention_model = Seq2SeqAttention(attn_enc, attn_dec, pad_idx=PAD_IDX).to(device)

    attn_history, attn_path = train_system(
        "Attention_LSTM",
        attention_model,
        train_loader,
        val_loader,
        epochs=epochs,
        lr=0.001,
        checkpoint_dir=checkpoint_dir,
        device=device
    )

    # Save combined training histories
    all_history = {
        "Baseline": base_history,
        "Attention": attn_history,
        "config": {
            "epochs": epochs,
            "batch_size": batch_size,
            "embed_dim": embed_dim,
            "hidden_dim": hidden_dim,
            "train_size": len(train_pairs),
            "val_size": len(val_pairs),
            "test_size": len(test_pairs),
            "src_vocab_size": len(src_vocab),
            "tgt_vocab_size": len(tgt_vocab)
        }
    }

    history_path = os.path.join(checkpoint_dir, "training_history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(all_history, f, indent=4)
    print(f"\nTraining complete! History saved to {history_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train NMT Baseline and Attention Models")
    parser.add_argument("--epochs", type=int, default=8, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--embed_dim", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension")
    args = parser.parse_args()

    main(
        epochs=args.epochs,
        batch_size=args.batch_size,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim
    )
