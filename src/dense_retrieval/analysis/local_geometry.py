"""Permutation-invariant query-relative geometry from corpus-local prototypes."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .prototype_correction import fit_spherical_kmeans


_EPS = 1e-12


def _normalized_rows(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or not array.shape[0] or not array.shape[1]:
        raise ValueError(f"{name} must be a non-empty two-dimensional array")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms <= _EPS):
        raise ValueError(f"{name} contains a zero-norm row")
    return array / norms


def prototype_counts(
    document_embeddings: np.ndarray,
    prototypes: np.ndarray,
    *,
    batch_size: int = 4096,
) -> np.ndarray:
    """Count nearest-prototype assignments without retaining document scores."""
    documents = _normalized_rows(document_embeddings, "document_embeddings")
    centers = _normalized_rows(prototypes, "prototypes")
    if documents.shape[1] != centers.shape[1]:
        raise ValueError("document_embeddings and prototypes must have equal width")
    if not isinstance(batch_size, (int, np.integer)) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    counts = np.zeros(len(centers), dtype=np.int64)
    for start in range(0, len(documents), int(batch_size)):
        labels = np.argmax(documents[start : start + batch_size] @ centers.T, axis=1)
        counts += np.bincount(labels, minlength=len(centers))
    return counts


@dataclass(frozen=True)
class CorpusPrototypeGeometry:
    """Document-only local prototype centers and their corpus occupancy."""

    prototypes: np.ndarray
    occupancy: np.ndarray

    def __post_init__(self) -> None:
        centers = _normalized_rows(self.prototypes, "prototypes")
        occupancy = np.asarray(self.occupancy, dtype=np.float64)
        if occupancy.shape != (len(centers),):
            raise ValueError("occupancy must have one value per prototype")
        if np.any(occupancy < 0) or not np.all(np.isfinite(occupancy)):
            raise ValueError("occupancy must be finite and non-negative")
        if occupancy.sum() <= _EPS:
            raise ValueError("occupancy must have positive mass")
        object.__setattr__(self, "prototypes", centers)
        object.__setattr__(self, "occupancy", occupancy / occupancy.sum())

    @classmethod
    def fit(
        cls,
        document_embeddings: np.ndarray,
        *,
        n_prototypes: int = 32,
        max_fit_documents: int = 20_000,
        max_iter: int = 30,
        random_state: int = 0,
        assignment_batch_size: int = 4096,
    ) -> "CorpusPrototypeGeometry":
        documents = _normalized_rows(document_embeddings, "document_embeddings")
        if not isinstance(n_prototypes, (int, np.integer)) or n_prototypes <= 0:
            raise ValueError("n_prototypes must be a positive integer")
        if max_fit_documents <= 0:
            raise ValueError("max_fit_documents must be positive")
        if len(documents) > max_fit_documents:
            rng = np.random.default_rng(random_state)
            fit_rows = np.sort(
                rng.choice(len(documents), size=max_fit_documents, replace=False)
            )
            fit_documents = documents[fit_rows]
        else:
            fit_documents = documents
        centers = fit_spherical_kmeans(
            fit_documents,
            n_clusters=min(int(n_prototypes), len(fit_documents)),
            max_iter=max_iter,
            random_state=random_state,
        )
        counts = prototype_counts(
            documents, centers, batch_size=assignment_batch_size
        )
        return cls(centers, counts)


def query_local_geometry_features(
    query_embeddings: np.ndarray,
    geometry: CorpusPrototypeGeometry,
    projection: np.ndarray,
    *,
    top_m: int = 8,
    temperature: float = 0.07,
) -> np.ndarray:
    """Aggregate local direction, variance, and density into invariant features.

    The result contains the posterior mean and diagonal variance of
    ``B.T (prototype - query)``, followed by six scalar neighborhood features:
    maximum similarity, posterior mean/std similarity, top-score gap,
    normalized posterior entropy, and selected prototype mass.
    """
    queries = _normalized_rows(query_embeddings, "query_embeddings")
    centers = geometry.prototypes
    basis = np.asarray(projection, dtype=np.float64)
    if basis.ndim != 2 or basis.shape[1] != queries.shape[1]:
        raise ValueError("projection must have shape (rank, embedding_dimension)")
    if centers.shape[1] != queries.shape[1]:
        raise ValueError("query_embeddings and prototypes must have equal width")
    if not isinstance(top_m, (int, np.integer)) or not 1 <= top_m <= len(centers):
        raise ValueError("top_m must be in [1, number of prototypes]")
    if temperature <= 0 or not np.isfinite(temperature):
        raise ValueError("temperature must be finite and positive")

    similarities = queries @ centers.T
    order = np.argsort(-similarities, axis=1, kind="mergesort")[:, :top_m]
    selected_scores = np.take_along_axis(similarities, order, axis=1)
    selected_centers = centers[order]
    selected_occupancy = geometry.occupancy[order]
    logits = selected_scores / float(temperature) + np.log(
        np.maximum(selected_occupancy, _EPS)
    )
    logits -= np.max(logits, axis=1, keepdims=True)
    weights = np.exp(logits)
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), _EPS)

    relative = (selected_centers - queries[:, None, :]) @ basis.T
    mean = np.sum(weights[..., None] * relative, axis=1)
    variance = np.sum(weights[..., None] * (relative - mean[:, None, :]) ** 2, axis=1)
    score_mean = np.sum(weights * selected_scores, axis=1)
    score_variance = np.sum(
        weights * (selected_scores - score_mean[:, None]) ** 2, axis=1
    )
    if top_m == 1:
        score_gap = np.zeros(len(queries), dtype=np.float64)
        entropy = np.zeros(len(queries), dtype=np.float64)
    else:
        score_gap = selected_scores[:, 0] - selected_scores[:, 1]
        entropy = -np.sum(weights * np.log(np.maximum(weights, _EPS)), axis=1)
        entropy /= np.log(float(top_m))
    scalars = np.column_stack(
        [
            selected_scores[:, 0],
            score_mean,
            np.sqrt(np.maximum(score_variance, 0.0)),
            score_gap,
            entropy,
            np.sum(selected_occupancy, axis=1),
        ]
    )
    return np.concatenate([mean, variance, scalars], axis=1).astype(np.float32)


__all__ = [
    "CorpusPrototypeGeometry",
    "prototype_counts",
    "query_local_geometry_features",
]
