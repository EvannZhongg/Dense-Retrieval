# Direct ranking-coordinate LOCO

The experiment fits a shared rank-32 ranking-boundary basis on training
corpora, optimizes privileged per-query ranking coordinates only for training
queries, and predicts those coordinates with matched Ridge models. The
`corpus_spectral` model adds query-conditioned interactions with a shared,
document-only spectral summary. Held-out qrels are used only for final metrics.

## Test NDCG@10

| held-out corpus | frozen | q-only | corpus spectral |
| --- | ---: | ---: | ---: |
| FiQA | 0.4484 | 0.4484 | 0.4484 |
| ArguAna | 0.3908 | 0.3905 | 0.3908 |
| SciFact | 0.8198 | 0.8057 | 0.1894 |
| NFCorpus | 0.3245 | 0.3063 | 0.1708 |
| mean | 0.4959 | 0.4877 | 0.3001 |

The corpus-spectral model has higher dev coordinate MSE than q-only in all four
folds and is never a reliable held-out policy. The severe SciFact and NFCorpus
failures show that document-spectrum interactions can extrapolate into harmful
coordinates even when their training objective is ranking-aligned.

This closes one important loophole: the earlier negative result cannot be
explained only by using nearest-positive deltas as the regression target.
Direct ranking-sensitive coordinate targets still do not yield a transferable
corpus-aware correction.
