# Privileged local-boundary benefit gating

## Question

The prototype probes may fail because their offline corpus summaries do not
resolve the documents that form the query's actual ranking boundary. Before
designing another deployable corpus representation, this experiment gives the
benefit gate a stronger, deliberately non-deployable view of that boundary.

For each query it performs a baseline Top-256 search and records the ranked
scores, document mean and standard deviation in the shared rank-32 correction
space at cutoffs 10/50/100/256, and the fixed query-only candidate's score
change on those documents. The raw query is retained, so the privileged model
nests the query-only input. This diagnostic violates the one-retrieval online
constraint and is not a proposed method. Its purpose is to test whether direct
unlabeled boundary observations make correction utility transferable.

All candidate directions, utility labels, ridge grids, dev selection, and
leave-one-corpus-out splits are unchanged from the earlier gate experiment.
Held-out qrels are used only for final utility and retrieval evaluation, never
to construct boundary features, fit a gate, or select a threshold.

## Retrieval results

The boundary gate beats frozen NDCG@10 in only 4 of 16 held-out folds and loses
in 12. Its four-model, equal-fold mean delta is -0.00065.

| Model | NDCG wins / losses vs frozen | Mean NDCG delta | Mean oracle-gate headroom |
| --- | ---: | ---: | ---: |
| BGE-M3 | 1 / 3 | -0.00357 | +0.01656 |
| E5 | 2 / 2 | +0.00307 | +0.02541 |
| Qwen3 | 1 / 3 | +0.00041 | +0.00704 |
| text-small | 0 / 4 | -0.00252 | +0.00870 |

E5/ArguAna accounts for the only large gain (HitRate@10 +0.05687 and NDCG@10
+0.03225). The same E5 gate loses NDCG on FiQA and SciFact, and the boundary
gate loses on ArguAna for all three other embedding models. This repeats the
isolated, non-transferable E5/ArguAna result from the prototype gate rather
than establishing a stable effect.

The boundary gate improves NDCG over the q-only gate in 10/16 folds, ties in
3, and loses in 3, for a mean difference of +0.00095. This reduction in harm
does not make it a useful policy because it still loses to frozen retrieval in
12/16 folds.

## Utility predictability

Direct prediction diagnostics also reject incremental utility information.

| Model | Mean utility NMSE | Mean correlation | Mean benefit AUC |
| --- | ---: | ---: | ---: |
| BGE-M3 | 1.552 | 0.045 | 0.558 |
| E5 | 2.152 | -0.009 | 0.429 |
| Qwen3 | 2.366 | 0.081 | 0.503 |
| text-small | 4.079 | -0.035 | 0.547 |

Every mean NMSE is worse than the constant held-out mean predictor, and the
correlations are near zero. Against the query-only gate, boundary features
improve NMSE in only 5/16 folds and lower mean benefit AUC by 0.0247. The small
retrieval advantage over learned controls therefore comes from threshold-level
abstention differences, not accurate transferable utility prediction.

## Conclusion

Even direct access to the baseline Top-256 neighborhood does not recover the
positive oracle-gating headroom under strict corpus holdout. Since this
privileged signal is richer than an admissible offline sketch and still fails,
there is no current evidence for investing in a more complex prototype,
covariance, or local-boundary surrogate. The reliable policy remains frozen
retrieval. Query-only correction and corpus-aware gating should remain
diagnostic research baselines until they show stable held-out benefit.
