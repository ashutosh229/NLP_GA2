import argparse, json, random
from collections import Counter
from common import *
from datasets import load_dataset


def toy_corpus(n, rng):
    names = ["Tom", "Mia", "Sam", "Lily", "Ben", "Anna"]
    things = ["ball", "kite", "box", "cake", "boat", "hat"]
    places = ["park", "garden", "forest", "beach", "house"]
    out = []
    for _ in range(n):
        a, t, p = rng.choice(names), rng.choice(things), rng.choice(places)
        adj = rng.choice(
            [
                "red",
                "big",
                "small",
                "shiny",
                "old",
                "soft",
                "blue",
                "funny",
                "warm",
                "new",
            ]
        )
        fr = rng.choice(
            ["Max", "Zoe", "Kim", "Joe", "Amy", "Dan", "Eva", "Leo", "Ivy", "Ray"]
        )
        out.append(
            f"{a} found a {adj} {t} in the {p}. {a} was very happy. "
            f"Then {a} showed the {t} to {fr}. They played all day and went home."
        )
    return out


def load_real(n_train, n_val, n_test, seed):
    ds = load_dataset("roneneldan/TinyStories")
    rng = random.Random(seed)
    tr = ds["train"]
    idx = rng.sample(range(len(tr)), int(n_train * 1.3))
    train_texts = tr.select(idx)["text"]
    held = ds["validation"]
    hidx = rng.sample(range(len(held)), int((n_val + n_test) * 1.5))
    held_texts = held.select(hidx)["text"]
    return train_texts, held_texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=20000)
    ap.add_argument("--n_val", type=int, default=1000)
    ap.add_argument("--n_test", type=int, default=1000)
    ap.add_argument("--min_freq", type=int, default=3)
    ap.add_argument("--max_vocab", type=int, default=10000)
    ap.add_argument("--min_story_tokens", type=int, default=30)
    ap.add_argument(
        "--max_story_tokens",
        type=int,
        default=300,
        help="longer stories are truncated (no <eos> added when truncated)",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--toy", action="store_true")
    a = ap.parse_args()
    set_seed(a.seed)
    rng = random.Random(a.seed)

    if a.toy:
        a.n_train, a.n_val, a.n_test, a.min_freq, a.min_story_tokens = (
            300,
            30,
            30,
            1,
            10,
        )
        train_texts = toy_corpus(a.n_train * 2, rng)
        held_texts = toy_corpus((a.n_val + a.n_test) * 2, rng)
    else:
        train_texts, held_texts = load_real(a.n_train, a.n_val, a.n_test, a.seed)

    def prep(texts, limit, forbidden):
        out, seen = [], set()
        for t in texts:
            c = clean(t)
            if c in seen or c in forbidden:
                continue
            toks = tokenize(t)
            if len(toks) < a.min_story_tokens:
                continue
            seen.add(c)
            out.append((t, toks))
            if len(out) == limit:
                break
        return out, seen

    train, train_seen = prep(train_texts, a.n_train, set())
    heldp, _ = prep(held_texts, a.n_val + a.n_test, train_seen)
    val, test = heldp[: a.n_val], heldp[a.n_val : a.n_val + a.n_test]
    print(f"stories  train={len(train)}  val={len(val)}  test={len(test)}")

    cnt = Counter(tok for _, toks in train for tok in toks[: a.max_story_tokens])
    words = [w for w, c in cnt.most_common() if c >= a.min_freq][
        : a.max_vocab - len(SPECIALS)
    ]
    vocab = Vocab(SPECIALS + words)
    vocab.save(DATA / "vocab.json")

    def to_ids(split):
        stories = []
        for _, toks in split:
            trunc = len(toks) > a.max_story_tokens
            ids = vocab.encode(toks[: a.max_story_tokens])
            stories.append([BOS_ID] + ids + ([] if trunc else [EOS_ID]))
        return stories

    splits = {"train": to_ids(train), "val": to_ids(val), "test": to_ids(test)}
    for k, v in splits.items():
        (DATA / f"{k}.json").write_text(json.dumps(v))
    (DATA / "test_texts.json").write_text(json.dumps([t for t, _ in test]))

    n_tok = sum(len(s) for s in splits["train"])
    n_unk = sum(s.count(UNK_ID) for s in splits["train"])
    meta = {
        "seed": a.seed,
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "vocab_size": len(vocab),
        "min_freq": a.min_freq,
        "max_vocab": a.max_vocab,
        "max_story_tokens": a.max_story_tokens,
        "train_tokens": n_tok,
        "train_unk_rate": n_unk / n_tok,
        "mean_story_len": n_tok / len(train),
        "tokenisation": "word-level, lower-case",
    }
    (DATA / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
