import csv, glob, json, itertools, statistics as st
from common import *

CRIT = ["coherence", "grammar", "relevance", "creativity"]


def main():
    key = json.loads((HUM / "key.json").read_text())
    files = sorted(glob.glob(str(HUM / "ratings_rater*.csv")))
    assert files, "no human_eval/ratings_rater*.csv files found"
    R = {}
    for f in files:
        name = Path(f).stem.replace("ratings_", "")
        R[name] = {}
        for row in csv.DictReader(open(f, encoding="utf-8")):
            R[name][row["sample_id"]] = {c: int(row[c]) for c in CRIT}
    raters = list(R)
    out = {
        "raters": raters,
        "by_model": {},
        "by_model_config": {},
        "by_rater": {},
        "agreement": {},
    }

    def collect(group_fn):
        g = {}
        for sid, meta in key.items():
            for r in raters:
                for c in CRIT:
                    g.setdefault(group_fn(meta), {}).setdefault(c, []).append(
                        R[r][sid][c]
                    )
        return {
            k: {c: {"mean": st.mean(v), "std": st.pstdev(v)} for c, v in d.items()}
            for k, d in g.items()
        }

    out["by_model"] = collect(lambda m: m["model"])
    out["by_model_config"] = collect(lambda m: f'{m["model"]}|{m["config"]}')
    for r in raters:
        out["by_rater"][r] = {c: st.mean(R[r][s][c] for s in key) for c in CRIT}
    for c in CRIT:
        diffs = [
            abs(R[a][s][c] - R[b][s][c])
            for a, b in itertools.combinations(raters, 2)
            for s in key
        ]
        out["agreement"][c] = {
            "mean_abs_diff": st.mean(diffs) if diffs else None,
            "exact_agree": sum(d == 0 for d in diffs) / len(diffs) if diffs else None,
            "within_1": sum(d <= 1 for d in diffs) / len(diffs) if diffs else None,
        }
    (RES / "human_eval_summary.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out["by_model"], indent=2))


if __name__ == "__main__":
    main()
