"""Small document-only spectral summaries for query-conditioned features."""
from __future__ import annotations

import numpy as np
from sklearn.utils.extmath import randomized_svd


_EPS = 1e-12


def fit_shared_document_basis(
    document_sets: list[np.ndarray],
    rank: int,
    *,
    max_documents_per_set: int = 5000,
    random_state: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a document-only centered basis on training corpora.

    The basis is shared across corpora.  Corpus-specific means and projected
    variances are computed separately afterwards, so a held-out corpus can be
    represented without any relevance labels.
    """
    if not document_sets:
        raise ValueError("at least one document set is required")
    if max_documents_per_set < 1:
        raise ValueError("max_documents_per_set must be positive")
    values = [np.asarray(documents, dtype=np.float32) for documents in document_sets]
    if any(doc.ndim != 2 or len(doc) == 0 for doc in values):
        raise ValueError("document sets must be non-empty 2-D arrays")
    dimension = values[0].shape[1]
    if any(doc.shape[1] != dimension for doc in values):
        raise ValueError("document sets must have the same feature width")
    if not 1 <= int(rank) <= dimension:
        raise ValueError("rank must be between 1 and the embedding dimension")
    sampled = []
    for documents in values:
        if len(documents) <= max_documents_per_set:
            sampled.append(documents)
            continue
        indices = np.linspace(
            0, len(documents) - 1, num=max_documents_per_set, dtype=np.int64
        )
        sampled.append(documents[indices])
    pooled = np.concatenate(sampled, axis=0)
    mean = pooled.mean(axis=0)
    centered = pooled - mean
    _, _, components = randomized_svd(
        centered,
        n_components=int(rank),
        n_iter=3,
        random_state=random_state,
    )
    return mean, components[: int(rank)]


def document_spectrum(
    documents: np.ndarray, mean: np.ndarray, components: np.ndarray
) -> dict[str, np.ndarray]:
    """Return document-only mean and diagonal projected covariance statistics."""
    values = np.asarray(documents, dtype=np.float32)
    center = np.asarray(mean, dtype=np.float32)
    basis = np.asarray(components, dtype=np.float32)
    if values.ndim != 2 or center.shape != (values.shape[1],):
        raise ValueError("documents and mean have incompatible shapes")
    if basis.ndim != 2 or basis.shape[1] != values.shape[1]:
        raise ValueError("components must be rank by embedding dimension")
    projected = (values - center) @ basis.T
    return {
        "mean_projection": projected.mean(axis=0),
        "variance": projected.var(axis=0),
        "log_variance": np.log(np.maximum(projected.var(axis=0), _EPS)),
        "spectral_mass": np.sum(projected * projected, axis=0),
    }


def query_spectral_interactions(
    queries: np.ndarray,
    document_stats: dict[str, np.ndarray],
    mean: np.ndarray,
    components: np.ndarray,
) -> np.ndarray:
    """Combine query projections with a corpus's precomputed spectral stats.

    The returned features contain only query-conditioned interactions.  No
    document/query score list or top-k result is materialized online.
    """
    values = np.asarray(queries, dtype=np.float32)
    center = np.asarray(mean, dtype=np.float32)
    basis = np.asarray(components, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != center.shape[0]:
        raise ValueError("queries and mean have incompatible shapes")
    if basis.ndim != 2 or basis.shape[1] != values.shape[1]:
        raise ValueError("components must be rank by embedding dimension")
    projected = (values - center) @ basis.T
    mean_projection = np.asarray(document_stats["mean_projection"], dtype=np.float32)
    variance = np.asarray(document_stats["variance"], dtype=np.float32)
    if mean_projection.shape != (len(basis),) or variance.shape != (len(basis),):
        raise ValueError("document statistics do not match components")
    return np.concatenate(
        [
            projected * mean_projection,
            projected * projected * variance,
            np.log1p(np.maximum(projected * projected * variance, 0.0)),
        ],
        axis=1,
    ).astype(np.float32)


__all__ = [
    "document_spectrum",
    "fit_shared_document_basis",
    "query_spectral_interactions",
]
