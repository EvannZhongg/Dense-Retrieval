# Prototype-cell moment benefit gating

## Hypothesis

Prototype centers and occupancy may miss the shape of the local document
decision boundary. This experiment adds document-only, per-prototype residual
means and diagonal variances projected into the rank-32 correction space. A
query aggregates the moments of its Top-8 prototype posterior. The fixed
reference control receives the same moments computed from training corpora.

All correction directions, gate models, hyperparameter grids, and LOCO splits
are otherwise identical to the plain text-small benefit-gating experiment.

## Results

| Held-out corpus | NDCG delta vs frozen | Delta vs plain corpus gate | Delta vs q-only gate | Delta vs reference gate |
| --- | ---: | ---: | ---: | ---: |
| FiQA | -0.0038 | -0.0010 | -0.0005 | -0.0020 |
| ArguAna | -0.0039 | 0.0000 | -0.0001 | -0.0025 |
| SciFact | -0.0032 | 0.0000 | 0.0000 | 0.0000 |
| NFCorpus | -0.0041 | -0.0006 | +0.0025 | +0.0011 |

The corpus-moment gate is below the frozen baseline on all four held-out
corpora. It never improves over the plain corpus gate. On training-corpus dev
selection, fixed-reference moments outperform corpus moments in three of four
folds, indicating that the added cell statistics primarily expose
non-transferable corpus differences.

## Conclusion

Within-prototype projected residual means and diagonal variances do not recover
the oracle benefit-gating headroom. The current prototype family has now tested
global occupancy, query-relative center direction, posterior uncertainty, and
within-cell first/second moments without stable held-out retrieval gain. Further
feature expansion within this family is not justified.
