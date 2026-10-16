import argparse, json, time, math
import torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from common import *
from models import RecurrentLM


def make_chunks(stories, seq_len):
    chunks = []
    for s in stories:
        for st in range(0, len(s) - 1, seq_len):
            c = s[st : st + seq_len + 1]
            if len(c) >= 2:
                chunks.append(c)
    return chunks


def collate(batch):
    L = max(len(c) for c in batch) - 1
    x = torch.full((len(batch), L), PAD_ID, dtype=torch.long)
    y = torch.full((len(batch), L), PAD_ID, dtype=torch.long)
    for i, c in enumerate(batch):
        x[i, : len(c) - 1] = torch.tensor(c[:-1])
        y[i, : len(c) - 1] = torch.tensor(c[1:])
    return x, y


def loader(split, seq_len, bs, shuffle):
    stories = json.loads((DATA / f"{split}.json").read_text())
    return DataLoader(
        make_chunks(stories, seq_len),
        batch_size=bs,
        shuffle=shuffle,
        collate_fn=collate,
    )


@torch.no_grad()
def evaluate(model, dl, device):
    model.eval()
    tot, n = 0.0, 0
    for x, y in dl:
        x, y = x.to(device), y.to(device)
        logits, _ = model(x)
        tot += F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
            ignore_index=PAD_ID,
            reduction="sum",
        ).item()
        n += (y != PAD_ID).sum().item()
    return tot / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", choices=["rnn", "lstm"], required=True)
    ap.add_argument("--emb", type=int, default=256)
    ap.add_argument("--hid", type=int, default=512)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--seq_len", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    set_seed(a.seed)
    dev = get_device()
    vocab = Vocab.load(DATA / "vocab.json")

    tr, va = loader("train", a.seq_len, a.bs, True), loader(
        "val", a.seq_len, a.bs, False
    )
    model = RecurrentLM(len(vocab), a.cell, a.emb, a.hid, a.layers, a.dropout).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=1)
    print(f"{a.cell}: {model.n_params():,} parameters on {dev}")

    hist = {"args": vars(a), "n_params": model.n_params(), "epochs": []}
    best, bad, t0 = float("inf"), 0, time.time()
    for ep in range(1, a.epochs + 1):
        model.train()
        t_ep, tot, n = time.time(), 0.0, 0
        for x, y in tr:
            x, y = x.to(dev), y.to(dev)
            logits, _ = model(x)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=PAD_ID
            )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)
            opt.step()
            k = (y != PAD_ID).sum().item()
            tot += loss.item() * k
            n += k
        train_loss, val_loss = tot / n, evaluate(model, va, dev)
        sched.step(val_loss)
        hist["epochs"].append(
            {
                "epoch": ep,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_ppl": math.exp(train_loss),
                "val_ppl": math.exp(val_loss),
                "epoch_seconds": time.time() - t_ep,
            }
        )
        print(
            f"ep {ep:2d} train {train_loss:.3f} val {val_loss:.3f} (ppl {math.exp(val_loss):.2f}) "
            f"{time.time()-t_ep:.0f}s"
        )
        if val_loss < best - 1e-4:
            best, bad = val_loss, 0
            torch.save(
                {"state": model.state_dict(), "cfg": model.cfg}, CKPT / f"{a.cell}.pt"
            )
        else:
            bad += 1
            if bad >= a.patience:
                print("early stopping")
                break
    hist["train_seconds"] = time.time() - t0

    model.load_state_dict(torch.load(CKPT / f"{a.cell}.pt", map_location=dev)["state"])
    hist["best_val_loss"] = best
    hist["val_ppl"] = math.exp(best)
    hist["test_loss"] = evaluate(model, loader("test", a.seq_len, a.bs, False), dev)
    hist["test_ppl"] = math.exp(hist["test_loss"])
    print(
        f"best val ppl {hist['val_ppl']:.2f} | test ppl {hist['test_ppl']:.2f} | "
        f"train time {hist['train_seconds']/60:.1f} min"
    )
    (RES / f"history_{a.cell}.json").write_text(json.dumps(hist, indent=2))


if __name__ == "__main__":
    main()
