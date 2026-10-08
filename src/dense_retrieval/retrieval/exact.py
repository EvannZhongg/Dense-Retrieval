from __future__ import annotations

import numpy as np


def exact_search(query_embeddings: np.ndarray, document_embeddings: np.ndarray, k: int = 10):
    """Return exact inner-product top-k indices and scores."""
    queries = np.asarray(query_embeddings, dtype=np.float32)
    documents = np.asarray(document_embeddings, dtype=np.float32)
    if queries.ndim != 2 or documents.ndim != 2 or queries.shape[1] != documents.shape[1]:
        raise ValueError(f"embedding shape mismatch: {queries.shape}, {documents.shape}")
    if int(k) < 1:
        raise ValueError("k must be positive")
    if len(documents) == 0:
        return np.empty((len(queries), 0), dtype=np.int64), np.empty((len(queries), 0), dtype=np.float32)
    k = max(1, min(int(k), len(documents)))
    scores = queries @ documents.T
    # A stable full sort keeps document input order as the deterministic tie-breaker.
    indices = np.argsort(-scores, axis=1, kind="stable")[:, :k]
    rows = np.arange(len(queries))[:, None]
    return indices, scores[rows, indices]
