import json
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import *

NAMES = {"rnn": "Vanilla RNN", "lstm": "LSTM"}


def main():
    H = {
        c: json.loads((RES / f"history_{c}.json").read_text()) for c in ("rnn", "lstm")
    }
    md = [
        "## Dataset\n",
        "```json",
        (DATA / "meta.json").read_text(),
        "```\n",
        "## Model comparison (Part 2 / 5)\n",
        "| Model | Params | Epochs run | Train time (min) | Best val loss | Val PPL | Test PPL |",
        "|---|---|---|---|---|---|---|",
    ]
    for c, h in H.items():
        md.append(
            f"| {NAMES[c]} | {h['n_params']:,} | {len(h['epochs'])} | {h['train_seconds']/60:.1f} | "
            f"{h['best_val_loss']:.3f} | {h['val_ppl']:.2f} | {h['test_ppl']:.2f} |"
        )

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for c, h in H.items():
        ep = [e["epoch"] for e in h["epochs"]]
        ax[0].plot(
            ep, [e["train_loss"] for e in h["epochs"]], "--", label=f"{NAMES[c]} train"
        )
        ax[0].plot(
            ep, [e["val_loss"] for e in h["epochs"]], "-", label=f"{NAMES[c]} val"
        )
        ax[1].plot(ep, [e["val_ppl"] for e in h["epochs"]], label=f"{NAMES[c]} val PPL")
    ax[0].set(title="Loss curves", xlabel="epoch", ylabel="cross-entropy (nats/token)")
    ax[1].set(title="Validation perplexity", xlabel="epoch", ylabel="perplexity")
    for x in ax:
        x.legend()
        x.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(RES / "loss_curves.png", dpi=150)

    gm = RES / "gen_metrics.json"
    if gm.exists():
        G = json.loads(gm.read_text())
        md += [
            "\n## Generation metrics on held-out test prompts (Part 5)\n",
            f"Settings: `{json.dumps(G['settings'])}`. Rep-3 = fraction of repeated trigram occurrences within a story; "
            "premature = EOS before 20 generated tokens; overlong = no EOS within budget.\n",
            "| Model | Decoding | Distinct-1 | Distinct-2 | Rep-3 | Avg len | Premature | Overlong | UNK rate |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for c, d in G["metrics"].items():
            for cfg, m in d.items():
                md.append(
                    f"| {NAMES[c]} | {cfg} | {m['distinct1']:.3f} | {m['distinct2']:.3f} | {m['rep3']:.3f} | "
                    f"{m['avg_len']:.1f} | {m['premature_rate']:.2%} | {m['overlong_rate']:.2%} | {m['unk_rate']:.2%} |"
                )
    hs = RES / "human_eval_summary.json"
    if hs.exists():
        S = json.loads(hs.read_text())
        md += [
            f"\n## Blinded human evaluation (raters: {', '.join(S['raters'])})\n",
            "| System | Coherence | Grammar | Relevance | Creativity |",
            "|---|---|---|---|---|",
        ]
        for k, d in S["by_model"].items():
            md.append(
                f"| {NAMES.get(k, k)} | "
                + " | ".join(
                    f"{d[c]['mean']:.2f} ± {d[c]['std']:.2f}"
                    for c in ("coherence", "grammar", "relevance", "creativity")
                )
                + " |"
            )
        md += [
            "\nInter-rater agreement:\n",
            "| Criterion | Mean abs diff | Exact | Within ±1 |",
            "|---|---|---|---|",
        ]
        for c, d in S["agreement"].items():
            md.append(
                f"| {c} | {d['mean_abs_diff']:.2f} | {d['exact_agree']:.0%} | {d['within_1']:.0%} |"
            )
    (RES / "tables.md").write_text("\n".join(md))
    print("wrote", RES / "tables.md", "and", RES / "loss_curves.png")


if __name__ == "__main__":
    main()
