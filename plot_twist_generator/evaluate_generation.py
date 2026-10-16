import argparse, json
from common import *
from generate import load_model, generate_ids
from metrics import summarise
from prompts import CONFIGS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_prompts", type=int, default=200)
    ap.add_argument("--prompt_len", type=int, default=20)
    ap.add_argument("--max_new", type=int, default=200)
    ap.add_argument("--seed", type=int, default=2024)
    a = ap.parse_args()
    dev = get_device()
    vocab = Vocab.load(DATA / "vocab.json")
    test = json.loads((DATA / "test.json").read_text())
    prompts = [s[1 : 1 + a.prompt_len] for s in test if len(s) > a.prompt_len + 10][
        : a.n_prompts
    ]
    print(f"{len(prompts)} test prompts of {a.prompt_len} tokens")

    out = {"settings": vars(a), "metrics": {}}
    for cell in ("rnn", "lstm"):
        model = load_model(cell, dev)
        out["metrics"][cell] = {}
        for cname, kw in CONFIGS.items():
            samples = []
            for i, p in enumerate(prompts):
                ids, ended = generate_ids(
                    model, p, dev, max_new=a.max_new, seed=a.seed + i, **kw
                )
                samples.append({"tokens": vocab.decode(ids), "ended": ended})
            m = summarise(samples, a.max_new)
            out["metrics"][cell][cname] = m
            print(cell, cname, {k: round(v, 3) for k, v in m.items()})
    (RES / "gen_metrics.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
