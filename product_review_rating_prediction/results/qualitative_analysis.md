# Qualitative Analysis — BiGRU vs. RoBERTa

Examples below are from the test set (raw text, as fed to both models). "BiGRU"/"RoBERTa"
predictions use each model's best seed (BiGRU seed 42, RoBERTa seed 123 — selected by
validation Macro-F1, consistent with the quantitative results).

## A. BiGRU fails, RoBERTa succeeds (5 examples)

**[1] True=3★ BiGRU=4★ RoBERTa=3★**
> "Good size desk for sewing with draws for storage of parts and thread... Not a bad unit for
> the price, **but** the quality is below Ikea... some of the draws did not [fit]."

*Hypothesis:* Classic contrastive/mixed-sentiment structure — positive opener ("good size",
"not a bad unit") followed by a "but"-pivot into concrete criticism. BiGRU's attention pooling
over the whole sequence still lets the early strong-positive tokens dominate; RoBERTa's
full self-attention better integrates the later qualifying clause and lands on the correct,
more moderate 3★.

**[2] True=3★ BiGRU=2★ RoBERTa=3★**
> "It was a great kit to put together **but** one gear is warped and i am finding myself
> **ripping my hair out** sanding and adjusting it..."

*Hypothesis:* The idiom "ripping my hair out" is figurative language with strongly negative
surface words ("ripping", "hair out") that a word-embedding+GRU model is likely to read
lexically as intensely negative, overshooting to 2★. RoBERTa's contextual embeddings handle
the idiom (and its co-occurrence with "great kit") more holistically, converging on the
correct mixed/neutral 3★.

**[3] True=2★ BiGRU=1★ RoBERTa=2★**
> "The story was all over the place. I felt no connection to Evelyn/Eva/Evie, and the ending
> was anticlimactic. **Thank goodness** it was a free book."

*Hypothesis:* Sarcasm ("thank goodness") plus negation ("no connection") and several
negative lexical cues stack up; BiGRU appears to sum/average these into an overly extreme
1★, while RoBERTa better calibrates the *magnitude* of negativity (clearly negative, but not
the most extreme class) — a case of miscalibrated intensity rather than wrong polarity.

**[4] True=2★ BiGRU=3★ RoBERTa=2★**
> "I was initially **very happy** with the 2 bowls... **However** after about 4 months, the
> inside of the bowls are covered with rust spots."

*Hypothesis:* Temporal sentiment shift via an explicit "however" pivot — strong early positive
("very happy") followed by a delayed negative outcome. BiGRU anchors on the strong early
positive span (3★); RoBERTa weighs the later, decisive negative clause correctly (2★). This is
the same contrastive-conjunction failure mode as example [1], mirrored in direction.

**[5] True=4★ BiGRU=3★ RoBERTa=4★**
> "Really enjoyed the movie, specially AFTER getting past all the really dark, dreary, scary
> stuff... Also the mommy issues... **wtf?!** ... I do feel misled.. this is a grown up movie
> in a kids wrapper..."

*Hypothesis:* A long, informal review with slang/interjections ("wtf?!"), ellipses, and
layered mixed sentiment (enjoyed overall, but several caveats). The informal tokens are likely
poorly covered by the fastText vocabulary (OOV → `<UNK>`) and the review is long enough that
BiGRU's fixed 200-token window and recurrent pooling lose some of the resolving context.
RoBERTa's byte-level BPE tokenizer degrades gracefully on slang and its pooled representation
better reconciles "really enjoyed" with "but... misled" into 4★.

**Pattern across [1], [2], [4], [5]:** these are exactly the cases the task highlights —
contrastive conjunctions ("but"/"however"), mixed sentiment, idiom/figurative language, and
long/informal text. BiGRU's recurrent+additive-attention pooling with frozen-then-finetuned
fastText embeddings is more easily anchored by locally strong sentiment words; RoBERTa's deep
self-attention over subword tokens integrates long-range contrastive structure more reliably.

## B. RoBERTa fails, BiGRU succeeds (2 examples)

**[1] True=4★ BiGRU=4★ RoBERTa=5★**
> "I got the Phone system on time and in good shape. It was what I ordered and in good shape."

*Hypothesis:* Short, repetitive, purely positive text with no contrast word or explicit
criticism — nothing in the text clearly distinguishes "4★ great" from "5★ perfect". This looks
like a case where the **gold label itself carries information not present in the text**
(the reviewer's internal bar for 5★ vs. 4★), rather than a genuine model error. RoBERTa's
strong positive-sentiment prior (fine-tuned to associate "good shape" ×2 + "what I ordered"
with very high ratings) overshoots to 5★; BiGRU's prediction matching the gold label here may
partly be luck given the ambiguity.

**[2] True=3★ BiGRU=3★ RoBERTa=2★**
> "...I thought I had better try this product instead. **It doesn't work any better than** any
> other cleaner made specifically for ceramic cooktops. So I'm..."

*Hypothesis:* A comparative, lukewarm verdict ("no better, no worse") rather than a strongly
negative one. RoBERTa appears to over-weight the negation pattern "doesn't work any better
than" as a strong negative cue (a known failure mode for fine-tuned transformers on
comparative negation, which is lexically negative but semantically closer to neutral),
pushing the prediction down to 2★. BiGRU happens to average out to the correct neutral 3★ here.

**Pattern:** RoBERTa's errors here are not about missing long-range structure (its usual
strength) but about (a) label ambiguity in very short, uniformly positive text, and (b)
over-sensitivity to negation phrasing in comparative sentences that are semantically mild.

## Token-level importance (gradient saliency), test idx 14

> "I mean the case looks cool. I don't know anything about functionality but they sent me a
> iPhone 8 Plus case. The title says 6 plus, I ordered a 6 plus and there was no option on
> selecting another iPhone size beside 6 and 6 plus so I'm **very disappointed** in the
> service" — True = 1★

See `token_importance.png`. The gradient-norm saliency (w.r.t. input embeddings, taken at the
true-class logit) concentrates on **"disappointed"**, **"no"** (as in "no option"), and the
product-mismatch tokens ("8 Plus" vs. "6 plus"), while the early mildly-positive opener ("looks
cool") receives comparatively low importance. This indicates RoBERTa correctly identifies the
decisive negative/complaint span late in the review rather than being anchored by the
early positive token — the same long-range-integration behavior that explains its advantage
over BiGRU in section A.

## Summary

- Per-class F1, overall Accuracy/Macro-F1/MAE, normalized confusion matrices, and the
  adjacent-vs-distant ordinal error breakdown are in `all_results.json`,
  `per_class_classification_report.json`, and `confusion_matrices.png`.
- Qualitatively, RoBERTa's advantage concentrates on reviews with **contrastive conjunctions,
  mixed sentiment, idioms, and long/informal text** — all cases needing long-range context
  integration that a fixed-window BiGRU handles less reliably.
- BiGRU's (rare) wins over RoBERTa cluster around **very short, label-ambiguous text** and
  **comparative-negation phrasing** that is lexically negative but semantically mild — i.e.
  cases where RoBERTa's stronger sentiment prior overshoots.
