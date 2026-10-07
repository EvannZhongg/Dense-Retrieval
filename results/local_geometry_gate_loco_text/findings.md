# Corpus-local benefit gating: text-embedding-3-small

## Protocol

- One fixed query-only nearest-positive ridge correction direction is shared by
  every gate.
- Candidate lambda is selected on training-corpus dev queries by the oracle
  gated NDCG@10 upper bound, deliberately giving gating a favorable test.
- Ridge utility gates predict per-query `corrected NDCG@10 - baseline NDCG@10`
  from matched q-only, fixed-reference, or corpus-local features.
- Gate alpha and threshold are selected on training-corpus dev NDCG@10.
- The held-out oracle gate uses test qrels for diagnosis only; it is never used
  for fitting or selection.

## Results

| Held-out | Baseline HR@10 | Corpus gate HR@10 | Oracle HR@10 | Baseline NDCG@10 | Corpus gate NDCG@10 | Oracle NDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FiQA | 0.7022 | 0.6991 | **0.7160** | 0.4484 | 0.4456 | **0.4662** |
| ArguAna | 0.8104 | 0.8009 | **0.8199** | 0.3908 | 0.3869 | **0.3985** |
| SciFact | 0.9111 | 0.9111 | 0.9111 | 0.8198 | 0.8167 | **0.8228** |
| NFCorpus | 0.6327 | 0.6327 | 0.6327 | 0.3245 | 0.3210 | **0.3310** |

The oracle gate proves that selective correction has useful headroom: it never
decreases a held-out metric and improves NDCG@10 in all four corpora. The
learned corpus gate fails to recover it. It lowers NDCG@10 in every corpus and
does not consistently beat either learned control. Its applied queries contain
many more neutral or harmful cases than beneficial cases.

## Conclusion

Corpus-local prototype geometry contains transferable information about the
nearest-positive direction, but not enough conditional information to predict
whether applying that direction helps the actual ranking. This separates two
hypotheses that earlier experiments conflated. The remaining bottleneck is
benefit prediction/generalization, not correction-space capacity or a missing
nonlinear model.
