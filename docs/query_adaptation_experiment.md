# Query-Side Adaptation Experiment

This experiment keeps the document vectors and the existing index unchanged.
The online operation is exactly one matrix multiplication on the query,
normalization, and the existing exact/ANN retrieval.

## Implemented hypotheses

* `gold`: a diagnostic reference. A ridge linear map learns to send a query
  embedding to the relevance-weighted centroid of its positive frozen document
  vectors. Training qrels must come from a declared train split and require a
  separate train-query embedding cache when IDs do not overlap with evaluation.
* `corpus`: a self-supervised alignment baseline. The same corpus text is
  encoded with the model's query protocol and mapped to its already cached
  document vector. No query qrels, evaluation queries, or document re-encoding
  are used. This is intentionally a minimal test of whether document-text
  alignment alone provides useful query-side supervision.

Both maps are stored as `weights` with shape `[query_dim, document_dim]` in an
`.npz` file plus a JSON provenance sidecar. The CLI refuses to run corpus mode
when the supplied model dimension differs from the frozen query cache.

## Reproduction

Run the infrastructure tests from the repository (the default Windows temp
directory may be inaccessible in restricted environments):

```text
pytest -q --basetemp tmp/pytest
```

Evaluate the existing frozen Qwen cache without re-encoding:

```text
python scripts/evaluate_baseline.py --cache cache/arguana/qwen3-embedding-0.6b/bec6f2759c73d88d509f --search-k 10
python scripts/evaluate_baseline.py --cache cache/fiqa/qwen3-embedding-0.6b/ca06d8c653000153fa02 --search-k 10
```

Gold mode requires a query cache for the train split if train and test IDs are
disjoint:

```text
python scripts/run_adaptation.py --cache <test-cache> --train-query-cache <train-cache> --mode gold --train-split train --eval-split test
```

For adaptation training, encode only the train queries; this avoids creating a
duplicate document cache:

```text
python scripts/encode_queries.py --dataset scifact --model qwen3-embedding-0.6b --split train --cache-root adaptation_cache
```

Corpus mode encodes corpus text through the query protocol, checkpointing that
additional query-side array under the adaptation output directory:

```text
python scripts/run_adaptation.py --cache <cache> --mode corpus --model <registry-name> --eval-split test
```

## Current evidence

The existing frozen baselines produced the following NDCG@10 values with exact
inner-product retrieval at search-k 10:

| Dataset | e5-base-v2 | bge-m3 | Qwen3-0.6B |
| --- | ---: | ---: | ---: |
| ArguAna (1,406 queries) | 0.3294 | 0.3977 | 0.4831 |
| FiQA (648 queries) | 0.3987 | 0.4130 | 0.4680 |

A complete `scifact/hash` end-to-end smoke run passed
for both modes, but hash embeddings are deliberately random and therefore are
only an integration check, not a retrieval result. Gold training was correctly
rejected when only a test query cache was supplied, preventing test-query
leakage.

The next valid research run is to create train-query caches with the same
frozen model/protocol as each existing document cache, then compare gold and
corpus alignment on held-out test qrels across multiple datasets. No positive
claim is made for the self-supervised method until those runs are complete.
