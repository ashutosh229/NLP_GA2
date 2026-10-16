from collections import Counter


def ngrams(seq, n):
    return [tuple(seq[i : i + n]) for i in range(len(seq) - n + 1)]


def distinct_n(list_of_token_lists, n):
    total, uniq = 0, set()
    for toks in list_of_token_lists:
        g = ngrams(toks, n)
        total += len(g)
        uniq.update(g)
    return len(uniq) / total if total else 0.0


def repetition_rate(toks, n=3):
    g = ngrams(toks, n)
    if not g:
        return 0.0
    return 1.0 - len(set(g)) / len(g)


def summarise(samples, max_new, premature_below=20, n_rep=3):
    toks = [s["tokens"] for s in samples]
    N = len(samples)
    lens = [len(t) for t in toks]
    return {
        "n_samples": N,
        "distinct1": distinct_n(toks, 1),
        "distinct2": distinct_n(toks, 2),
        "rep3": sum(repetition_rate(t, n_rep) for t in toks) / max(N, 1),
        "avg_len": sum(lens) / max(N, 1),
        "premature_rate": sum(
            1 for s in samples if s["ended"] and len(s["tokens"]) < premature_below
        )
        / max(N, 1),
        "overlong_rate": sum(1 for s in samples if not s["ended"]) / max(N, 1),
        "unk_rate": sum(t.count("<unk>") for t in toks) / max(sum(lens), 1),
    }
