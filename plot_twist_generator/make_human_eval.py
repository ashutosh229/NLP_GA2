import json, random, csv, argparse
from common import *


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--include_llm", action="store_true")
    a = ap.parse_args()
    rows = [
        r
        for r in json.loads((RES / "generations_part4.json").read_text())
        if r["sample"] == 0
    ]
    llm_path = RES / "llm_part4.json"
    if a.include_llm and llm_path.exists():
        rows += [r for r in json.loads(llm_path.read_text()) if r["sample"] == 0]
    random.Random(a.seed).shuffle(rows)
    key, out = {}, []
    for i, r in enumerate(rows, 1):
        sid = f"S{i:02d}"
        key[sid] = {
            "model": r["model"],
            "config": r["config"],
            "prompt_id": r["prompt_id"],
            "seed": r["seed"],
        }
        out.append(
            {"sample_id": sid, "prompt": r["prompt"], "continuation": r["continuation"]}
        )
    (HUM / "key.json").write_text(json.dumps(key, indent=1))
    with open(HUM / "ratings_template.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "sample_id",
                "prompt",
                "continuation",
                "coherence",
                "grammar",
                "relevance",
                "creativity",
            ],
        )
        w.writeheader()
        for r in out:
            w.writerow(
                {**r, "coherence": "", "grammar": "", "relevance": "", "creativity": ""}
            )
    (HUM / "INSTRUCTIONS.md").write_text(
        "# Rater instructions\nYou will read short story continuations. You are NOT told which system wrote them.\n"
        "For each row, give integer scores 1 (very poor) - 5 (excellent):\n"
        "- **coherence**: do events/entities stay consistent and make sense together?\n"
        "- **grammar**: grammatical quality of the sentences.\n"
        "- **relevance**: does it follow from the opening prompt (same characters, situation)?\n"
        "- **creativity**: is there an interesting development or surprise, rather than filler/repetition?\n"
        "Copy `ratings_template.csv` to `ratings_rater1.csv`, `ratings_rater2.csv`, ... (one per rater, "
        "work independently), fill the four score columns, and do not edit other columns. Do not discuss with other raters.\n"
    )
    print(
        f"{len(out)} blinded samples -> {HUM}. Give each rater a copy of ratings_template.csv ONLY."
    )


if __name__ == "__main__":
    main()
