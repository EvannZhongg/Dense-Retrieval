# Corpus-local coordinate probe across embedding models

## Protocol

The text-small LOCO probe was repeated without changing the representation or
hyperparameter grids for Qwen3-Embedding-0.6B, BGE-M3, and E5-Base-v2. Every
model uses four held-out corpora (FiQA, ArguAna, SciFact, NFCorpus), a rank-32
zero-origin correction space, 16 document-only corpus prototypes, and Top-8
query-relative prototype features.

## Coordinate prediction

Corpus-local geometry improved normalized MSE over both controls in every one
of the 12 new held-out folds.

| Model | Wins over q-only | Wins over reference | Mean NMSE gain over q-only | Mean gain over reference |
| --- | ---: | ---: | ---: | ---: |
| Qwen3 | 4/4 | 4/4 | 0.0491 | 0.0741 |
| BGE-M3 | 4/4 | 4/4 | 0.0732 | 0.0643 |
| E5 | 4/4 | 4/4 | 0.0747 | 0.0755 |

Together with text-embedding-3-small, corpus-local geometry wins 16/16 folds
against both matched controls. This is strong evidence that the frozen corpus
contains transferable incremental information about the nearest-positive
correction direction beyond the query embedding alone.

## Retrieval

The coordinate gain still does not imply stable retrieval improvement. Examples
include BGE-M3/NFCorpus HitRate@10 increasing from 0.5306 to 0.5510 and
E5/ArguAna increasing from 0.6682 to 0.6777, while BGE-M3/FiQA decreases from
0.6806 to 0.6728 and Qwen3/ArguAna decreases from 0.9005 to 0.8957. Most
NDCG@10 changes are negative, and corpus geometry does not consistently beat
the fixed-reference feature control.

## Conclusion

The original broad hypothesis must be split. Query-conditioned corpus geometry
does provide cross-model, cross-corpus incremental direction information, but
the nearest-positive target is not a reliable surrogate for retrieval utility.
The remaining question is whether a corpus representation can describe the
local ranking boundary well enough to predict when and how far to move.
