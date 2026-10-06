from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

from ..retrieval.exact import exact_search
from .failures import add_failure_groups
from .geometry import compute_geometry_metrics
from .quality import DEFAULT_CUTOFFS, compute_quality_metrics


def evaluate_embeddings(
    query_ids: Sequence[str],
    qrels: Dict[str, Dict[str, int]],
    query_embeddings: np.ndarray,
    document_embeddings: np.ndarray,
    document_ids: Sequence[str],
    temperature: float = 1.0,
    search_k: int = 100,
):
    required_k = max(max(DEFAULT_CUTOFFS), int(search_k))
    ranked_indices, ranked_scores = exact_search(
        query_embeddings, document_embeddings, required_k
    )
    ranked_document_ids = np.asarray(document_ids, dtype=object)[ranked_indices].tolist()

    aggregate, quality = compute_quality_metrics(
        query_ids,
        qrels,
        ranked_document_ids,
        DEFAULT_CUTOFFS,
        set(map(str, document_ids)),
    )
    geometry = compute_geometry_metrics(
        query_ids,
        qrels,
        query_embeddings,
        document_embeddings,
        document_ids,
        ranked_scores,
        temperature,
    )
    per_query = quality.merge(
        geometry, on="query_id", how="inner", validate="one_to_one"
    )
    return aggregate, add_failure_groups(per_query)
