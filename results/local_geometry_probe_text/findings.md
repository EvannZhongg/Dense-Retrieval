# Corpus-local geometry probe: text-embedding-3-small

## Protocol

- Strict leave-one-corpus-out over FiQA, ArguAna, SciFact, and NFCorpus.
- Frozen query/document embeddings and unchanged document indices.
- Zero-origin rank-32 correction basis fitted on training corpora only.
- Sixteen document-only spherical prototypes per corpus; Top-8 posterior features.
- Ridge alpha and correction lambda selected by macro validation performance on
  training corpora only.
- Held-out corpus qrels used only for final test target and retrieval metrics.

`reference_geometry` uses the same feature construction and dimensionality as
`corpus_geometry`, but its prototypes are fitted once from the training corpora.
It controls for a nonlinear transformation of the query that contains no
held-out corpus information.

## Coordinate prediction

Lower normalized MSE is better.

| Held-out corpus | q_only | reference_geometry | corpus_geometry |
| --- | ---: | ---: | ---: |
| FiQA | 1.0873 | 1.1351 | **1.0218** |
| ArguAna | 1.4104 | 1.4379 | **1.3789** |
| SciFact | 0.6282 | 0.6303 | **0.6023** |
| NFCorpus | 0.5463 | 0.5476 | **0.5055** |

Corpus-local geometry improves oracle-coordinate prediction over both controls
in all four held-out corpora. The gain is therefore not explained solely by
adding a query-derived nonlinear feature map. Absolute transfer remains weak on
FiQA and ArguAna, where variance-weighted R2 is negative.

## Retrieval

Held-out HitRate@10 after training-corpus validation selects lambda:

| Held-out corpus | baseline | q_only | reference_geometry | corpus_geometry |
| --- | ---: | ---: | ---: | ---: |
| FiQA | 0.7022 | 0.6991 | **0.7052** | **0.7052** |
| ArguAna | **0.8104** | 0.8057 | 0.8057 | 0.8057 |
| SciFact | 0.9111 | 0.9111 | 0.9111 | 0.9111 |
| NFCorpus | 0.6327 | 0.6327 | 0.6327 | 0.6327 |

The coordinate-prediction gain does not translate into a corpus-specific
retrieval gain. FiQA improves slightly over the frozen baseline, but the
query-only reference control reaches the same HitRate@10. ArguAna degrades;
SciFact and NFCorpus are unchanged at HitRate@10, with several secondary
ranking metrics lower than baseline.

## Conclusion

Corpus-local prototype geometry contains a small, transferable signal about
the nearest-positive correction coordinates. This is a useful positive result
for the representation question, but it is not yet an effective correction
method. The current nearest-positive regression target is insufficiently
aligned with ranking, and a larger predictor is not justified by these results.
The next falsifiable step is a simple ranking-aligned linear or gated model that
uses the same fixed local-geometry features and must beat both frozen retrieval
and the matched query-only control under the same LOCO protocol.
