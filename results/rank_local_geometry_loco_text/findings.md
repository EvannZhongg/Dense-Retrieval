# Linear ranking-aligned local geometry: text-embedding-3-small

## Protocol

- Strict leave-one-corpus-out over FiQA, ArguAna, SciFact, and NFCorpus.
- Frozen embedding model, document embeddings, and document index.
- Rank-32 zero-origin correction subspace fitted on training corpora only.
- One zero-initialized linear map trained with multi-positive hard-negative
  ranking loss and corpus-balanced updates.
- Matched `rank_q_only`, `rank_reference_geometry`, and
  `rank_corpus_geometry` inputs; lambda selected by macro training-corpus dev
  HitRate@10.
- Held-out qrels used for final test evaluation only.

## Results

| Held-out corpus | Baseline HR@10 | Corpus HR@10 | Baseline NDCG@10 | Corpus NDCG@10 | Selected lambda |
| --- | ---: | ---: | ---: | ---: | ---: |
| FiQA | 0.7022 | 0.6991 | 0.4484 | 0.4404 | 0.05 |
| ArguAna | 0.8104 | 0.8104 | 0.3908 | 0.3908 | 0.00 |
| SciFact | 0.9111 | 0.9111 | 0.8198 | 0.8094 | 0.05 |
| NFCorpus | 0.6327 | 0.6327 | 0.3245 | 0.3245 | 0.00 |

Corpus geometry produced the lowest training ranking loss in every fold, but
that small advantage did not transfer. Two folds selected exact abstention;
the two non-zero folds lost NDCG, and FiQA also lost HitRate@10. Directly
aligning a linear correction with ranking therefore does not rescue the local
geometry signal observed in nearest-positive coordinate prediction.

## Conclusion

The negative retrieval result is not specific to the nearest-positive
regression target. A ranking-aligned linear model can exploit corpus features
in-sample, but the improvement is not transferable to an unseen corpus. More
model capacity is not justified by this evidence.
