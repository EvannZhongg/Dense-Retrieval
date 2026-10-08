from __future__ import annotations

from ..adaptation import LinearQueryCalibrator
from .baseline import evaluate_baseline


def evaluate_calibrator(
    dataset,
    query_embeddings,
    document_embeddings,
    document_ids,
    calibrator: LinearQueryCalibrator,
    search_k: int = 10,
):
    """Evaluate one calibrated query pass against the unchanged document index."""
    return evaluate_baseline(
        dataset,
        calibrator.transform(query_embeddings),
        document_embeddings,
        document_ids,
        search_k,
    )
