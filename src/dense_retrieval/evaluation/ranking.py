"""Top-k ranking and the metric subset reported by the correction studies.

``evaluate_retrieval`` scores one query set against a fixed corpus.  It is the
cheap, per-configuration counterpart of
:func:`dense_retrieval.evaluation.evaluate_embeddings`, which additionally
computes geometry and failure groups but cannot be reused for corrected query
embeddings that no longer correspond to stored query ids.
"""
from __future__ import annotations

import numpy as np

from ..retrieval.exact import exact_search
from .quality import compute_quality_metrics

METRICS = [
    "HitRate@5",
    "HitRate@10",
    "Recall@5",
    "Recall@10",
    "MRR@10",
    "NDCG@10",
]


def top_k_document_ids(
    query_embeddings, document_embeddings, document_ids, k=10, batch_size=128
):
    """Return the top-``k`` document ids per query, batched over queries."""
    rankings = []
    document_ids = np.asarray(document_ids, dtype=object)
    for start in range(0, len(query_embeddings), batch_size):
        indices, _ = exact_search(
            query_embeddings[start : start + batch_size], document_embeddings, k
        )
        rankings.extend(document_ids[indices].tolist())
    return rankings


def evaluate_retrieval(dataset, queries, documents, batch_size=128):
    """Return :data:`METRICS` for ``queries`` ranked against ``documents``."""
    rankings = top_k_document_ids(
        queries, documents, list(dataset.corpus), 10, batch_size
    )
    metrics, _ = compute_quality_metrics(
        [sample.query_id for sample in dataset.queries],
        dataset.qrels,
        rankings,
        cutoffs=(5, 10),
        available_document_ids=set(dataset.corpus),
    )
    return {metric: metrics[metric] for metric in METRICS}
