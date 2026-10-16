import argparse, json, torch
from common import *
from models import RecurrentLM
from prompts import *


def load_model(cell, device):
    ck = torch.load(CKPT / f"{cell}.pt", map_location=device)
    m = RecurrentLM(**ck["cfg"]).to(device)
    m.load_state_dict(ck["state"])
    return m.eval()


def pick_next(logits, strategy, temperature, top_k, gen):
    logits = logits.float().cpu().clone()
    logits[PAD_ID] = float("-inf")
    logits[BOS_ID] = float("-inf")
    if strategy == "greedy":
        return int(logits.argmax())
    logits = logits / temperature
    if top_k and top_k > 0:
        kth = torch.topk(logits, top_k).values[-1]
        logits[logits < kth] = float("-inf")
    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, 1, generator=gen))


@torch.no_grad()
def generate_ids(
    model,
    prompt_ids,
    device,
    strategy="greedy",
    temperature=1.0,
    top_k=0,
    max_new=200,
    seed=0,
):
    gen = torch.Generator().manual_seed(seed)
    x = torch.tensor([[BOS_ID] + prompt_ids], device=device)
    logits, h = model(x)
    out, ended = [], False
    for _ in range(max_new):
        nxt = pick_next(logits[0, -1], strategy, temperature, top_k, gen)
        if nxt == EOS_ID:
            ended = True
            break
        out.append(nxt)
        logits, h = model(torch.tensor([[nxt]], device=device), h)
    return out, ended


def run_prompt_set(models, vocab, prompts, configs, n_samples, max_new, base_seed, dev):
    rows = []
    for pid, text in prompts.items():
        p_ids = vocab.encode(tokenize(text))
        for cell, model in models.items():
            for cname in configs:
                kw = CONFIGS[cname]
                ns = 1 if kw["strategy"] == "greedy" else n_samples
                for s in range(ns):
                    seed = base_seed + s
                    ids, ended = generate_ids(
                        model, p_ids, dev, max_new=max_new, seed=seed, **kw
                    )
                    toks = vocab.decode(ids)
                    rows.append(
                        {
                            "prompt_id": pid,
                            "prompt": text,
                            "model": cell,
                            "config": cname,
                            "sample": s,
                            "seed": seed,
                            "tokens": toks,
                            "ended": ended,
                            "continuation": detokenize(toks),
                        }
                    )
    return rows


def to_markdown(rows, title):
    md = [
        f"# {title}\n",
        "_Everything inside a `MODEL OUTPUT` block is raw, unedited model text. "
        "Commentary is written by the authors._\n",
    ]
    for r in rows:
        md += [
            f"### [{r['prompt_id']}] model={r['model']} | decoding={r['config']} | seed={r['seed']} "
            f"| tokens={len(r['tokens'])} | ended={r['ended']}",
            f"**Prompt (human-written):** {r['prompt']}\n",
            "```text MODEL OUTPUT",
            r["continuation"],
            "```",
            "**Commentary:** _(write your analysis here)_\n",
        ]
    return "\n".join(md)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_samples", type=int, default=3)
    ap.add_argument("--max_new", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args()
    dev = get_device()
    vocab = Vocab.load(DATA / "vocab.json")
    models = {c: load_model(c, dev) for c in ("rnn", "lstm")}

    p3 = run_prompt_set(
        models, vocab, PART3_PROMPTS, list(CONFIGS), a.n_samples, a.max_new, a.seed, dev
    )
    p4 = run_prompt_set(
        models, vocab, PART4_PROMPTS, PART4_CONFIGS, 2, a.max_new, a.seed, dev
    )
    (RES / "generations_part3.json").write_text(json.dumps(p3, indent=1))
    (RES / "generations_part4.json").write_text(json.dumps(p4, indent=1))
    (RES / "generated_stories_part3.md").write_text(
        to_markdown(p3, "Part 3 - decoding strategies")
    )
    (RES / "generated_stories_part4.md").write_text(
        to_markdown(p4, "Part 4 - Plot-Twist Challenge")
    )
    print(f"part3: {len(p3)} samples, part4: {len(p4)} samples -> {RES}")


if __name__ == "__main__":
    main()
