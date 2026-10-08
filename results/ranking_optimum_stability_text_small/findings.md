# Ranking-optimum stability audit

This is a privileged identifiability diagnostic, not a deployable correction result. A high retrieval overlap paired with low coordinate cosine is direct evidence that coordinate regression has no unique target under the current ranking objective. Directionally distinct means cosine below 0.2; metric equivalence requires NDCG@10 and MRR@10 differences at most 0.01, and ranking equivalence additionally requires Top-10 overlap of at least 0.8.

| comparison_group | coordinate_cosine | coordinate_cosine_p10 | top10_overlap | abs_ndcg_difference | abs_mrr_difference | abs_reference_rank_loss_difference | directionally_distinct_fraction | distinct_but_metric_equivalent_fraction | distinct_but_ranking_equivalent_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| initialization | 0.9798 | 0.9857 | 0.9519 | 0.0090 | 0.0135 | 0.0038 | 0.0096 | 0.0022 | 0.0000 |
| negative_count | 0.9339 | 0.8801 | 0.6669 | 0.0738 | 0.0806 | 0.0703 | 0.0093 | 0.0014 | 0.0000 |
| negative_sample | 0.9043 | 0.8142 | 0.6718 | 0.0611 | 0.0650 | 0.1230 | 0.0163 | 0.0034 | 0.0000 |
| optimizer_seed | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| regularization | 0.9973 | 0.9935 | 0.9150 | 0.0163 | 0.0139 | 0.0074 | 0.0000 | 0.0000 | 0.0000 |
