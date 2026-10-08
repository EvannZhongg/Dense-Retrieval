# Action-space audit

This is a privileged capacity diagnostic. Every basis is fitted without the
held-out corpus, but projection coefficients and lambda selection use held-out
qrels only to measure the best utility available inside each action space.

## Rank 32 projected nearest-positive oracle

| method | mean_ndcg_gain_retention | mean_best_ndcg | mean_oracle_lambda |
| --- | --- | --- | --- |
| ranking_boundary_svd | 0.3109 | 0.5892 | 1.0000 |
| delta_svd | 0.2859 | 0.5838 | 1.0000 |
| tangent_delta_svd | 0.2832 | 0.5822 | 1.0000 |
| random_subspace | 0.0383 | 0.5097 | 1.0000 |

## Rank 32 direct ranking oracle

| method | mean_ranking_oracle_gain_retention | mean_ranking_oracle_ndcg |
| --- | --- | --- |
| random_subspace | 0.8234 | 0.7928 |
| ranking_boundary_svd | 0.7848 | 0.7923 |
| delta_svd | 0.7629 | 0.7865 |
| tangent_delta_svd | 0.7549 | 0.7810 |

## Interpretation

The action-space hypothesis is not falsified. With held-out qrels available to
optimize coordinates, rank 32 subspaces retain substantial ranking headroom;
rank 64 retains about 85% of the full-ambient oracle gain.  Therefore the
earlier failure of query-only and corpus-aware predictors cannot be explained
by PCA rank 32 having no ranking capacity.

The ranking-boundary basis is directionally better than delta SVD, but the
difference is small and not uniform across folds.  The random-subspace result
is a capacity control, not a candidate method: it demonstrates that per-query
coordinate optimization can exploit many directions that a learned predictor
does not know how to select.

The next experiment should therefore predict or gate direct ranking-sensitive
coordinates in a query-dependent basis, while retaining frozen retrieval,
single-pass correction, and strict leave-one-corpus-out selection.  No online
method improvement is claimed by this audit.
