from __future__ import annotations

from ..retrieval import exact_search
from .metrics import compute_metrics


def evaluate_baseline(dataset, query_embeddings, document_embeddings, document_ids, search_k=10):
    if int(search_k) < 1:
        raise ValueError("search_k must be positive")
    indices, _ = exact_search(query_embeddings, document_embeddings, max(int(search_k), 10))
    ids = list(document_ids)
    rankings = [[ids[int(index)] for index in row] for row in indices]
    return compute_metrics([sample.query_id for sample in dataset.queries], dataset.qrels, rankings, cutoffs=(5, 10))
