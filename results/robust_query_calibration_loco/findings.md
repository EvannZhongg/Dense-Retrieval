# Robust query-only lambda selection

## Hypothesis

The query-only correction direction may be useful even if ordinary macro-dev
lambda selection overfits corpus-specific retrieval changes. This experiment
tests a conservative cross-corpus rule: a nonzero lambda is eligible only when
its dev HitRate@10 is non-decreasing on every training corpus and its macro dev
HitRate@10 strictly improves over exact `lambda=0`. Selection and model fitting
remain leave-one-corpus-out; held-out qrels are used only for final evaluation.

The unchanged rank-32 local-geometry probe is run for text-embedding-3-small,
Qwen3, BGE-M3, and E5 on FiQA, ArguAna, SciFact, and NFCorpus. The complete
per-corpus dev curves are in `local_geometry_lambda_selection.csv`.

## Query-only results

The rule selected a nonzero query-only correction in 15 of 16 held-out folds.
It did not make the correction safe on an unseen corpus.

| Model | Nonzero folds | HitRate@10 wins / ties / losses | Mean HitRate@10 delta | NDCG@10 wins / ties / losses | Mean NDCG@10 delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| BGE-M3 | 4/4 | 2 / 0 / 2 | -0.00043 | 2 / 0 / 2 | -0.00002 |
| E5 | 3/4 | 1 / 3 / 0 | +0.00237 | 1 / 1 / 2 | -0.00192 |
| Qwen3 | 4/4 | 1 / 2 / 1 | -0.00160 | 0 / 1 / 3 | -0.00191 |
| text-small | 4/4 | 0 / 2 / 2 | -0.00196 | 1 / 0 / 3 | -0.00170 |

The failure is not specific to the query-only feature set. Under the same
selection rule, corpus-local geometry has negative mean NDCG@10 for all four
embedding models and fixed-reference geometry has negative mean NDCG@10 for
all four models. Corpus-aware features therefore do not recover a transferable
step-size policy here.

## Why the rule fails

HitRate@10 is discrete, so a harmful direction can tie the weakest training
corpus while improving another. The non-decrease condition consequently admits
many nonzero lambdas without establishing a positive margin.

Requiring a strictly positive HitRate@10 gain on every training corpus would
leave a nonzero candidate in only 2 of 16 query-only folds. Both candidates are
for held-out FiQA, and both lose there: BGE-M3 changes HitRate@10 by -0.00463
and NDCG@10 by -0.00563, while text-small changes them by -0.00309 and
-0.00160. The stricter rule would therefore be mostly baseline abstention and
still would not certify its two corrections.

## Conclusion

Cross-corpus dev non-degradation is insufficient for held-out safety. The
available query-only correction is not a reliable deployment policy, and a
strict all-corpus positive rule mostly collapses to the frozen baseline without
making the remaining nonzero decisions trustworthy. Further lambda-selection
heuristics are not justified unless a new observable predicts held-out ranking
utility rather than merely fitting discrete dev metrics.
