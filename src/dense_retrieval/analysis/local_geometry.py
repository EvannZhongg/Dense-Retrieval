"""Permutation-invariant query-relative geometry from corpus-local prototypes."""
from __future__ import annotations

from collections.abc import Mapping
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


@dataclass(frozen=True)
class ProjectedCellMoments:
    """Per-prototype residual mean and diagonal variance in one projection."""

    mean: np.ndarray
    variance: np.ndarray

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean, dtype=np.float64)
        variance = np.asarray(self.variance, dtype=np.float64)
        if mean.ndim != 2 or variance.shape != mean.shape:
            raise ValueError("mean and variance must have equal two-dimensional shape")
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(variance)):
            raise ValueError("cell moments must be finite")
        if np.any(variance < 0):
            raise ValueError("cell variance must be non-negative")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "variance", variance)


def projected_cell_moments(
    document_embeddings: np.ndarray | Mapping[str, np.ndarray],
    geometry: CorpusPrototypeGeometry,
    projection: np.ndarray,
    *,
    max_documents: int = 5000,
    batch_size: int = 1024,
    random_state: int = 0,
) -> ProjectedCellMoments:
    """Estimate within-cell residual moments from a document-only sample."""
    basis = np.asarray(projection, dtype=np.float64)
    if basis.ndim != 2 or basis.shape[1] != geometry.prototypes.shape[1]:
        raise ValueError("projection must have shape (rank, embedding_dimension)")
    if max_documents <= 0 or batch_size <= 0:
        raise ValueError("max_documents and batch_size must be positive")
    if isinstance(document_embeddings, Mapping):
        if not document_embeddings:
            raise ValueError("document_embeddings must not be empty")
        sources = [document_embeddings[name] for name in sorted(document_embeddings)]
    else:
        sources = [document_embeddings]
    rng = np.random.default_rng(random_state)
    per_source = max(1, int(max_documents) // len(sources))
    sampled = []
    for source in sources:
        values = np.asarray(source)
        if values.ndim != 2 or values.shape[1] != basis.shape[1] or not len(values):
            raise ValueError("document embeddings have invalid shape")
        size = min(per_source, len(values))
        rows = np.sort(rng.choice(len(values), size=size, replace=False))
        sampled.append(values[rows])

    count = np.zeros(len(geometry.prototypes), dtype=np.int64)
    total = np.zeros((len(geometry.prototypes), len(basis)), dtype=np.float64)
    total_square = np.zeros_like(total)
    centers = geometry.prototypes
    projected_centers = centers @ basis.T
    for values in sampled:
        for start in range(0, len(values), int(batch_size)):
            batch = np.asarray(values[start : start + batch_size], dtype=np.float64)
            batch /= np.maximum(np.linalg.norm(batch, axis=1, keepdims=True), _EPS)
            labels = np.argmax(batch @ centers.T, axis=1)
            residual = batch @ basis.T - projected_centers[labels]
            count += np.bincount(labels, minlength=len(centers))
            np.add.at(total, labels, residual)
            np.add.at(total_square, labels, residual**2)
    denominator = np.maximum(count[:, None], 1)
    mean = total / denominator
    variance = np.maximum(total_square / denominator - mean**2, 0.0)
    return ProjectedCellMoments(mean, variance)


def query_local_geometry_features(
    query_embeddings: np.ndarray,
    geometry: CorpusPrototypeGeometry,
    projection: np.ndarray,
    *,
    top_m: int = 8,
    temperature: float = 0.07,
    cell_moments: ProjectedCellMoments | None = None,
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
    parts = [mean, variance, scalars]
    if cell_moments is not None:
        if cell_moments.mean.shape != (len(centers), basis.shape[0]):
            raise ValueError(
                "cell moments must have shape (number_of_prototypes, projection_rank)"
            )
        selected_mean = cell_moments.mean[order]
        selected_variance = cell_moments.variance[order]
        parts.extend(
            [
                np.sum(weights[..., None] * selected_mean, axis=1),
                np.sum(
                    weights[..., None]
                    * np.sqrt(np.maximum(selected_variance, 0.0)),
                    axis=1,
                ),
            ]
        )
    return np.concatenate(parts, axis=1).astype(np.float32)


def fit_reference_geometry(
    document_embeddings: Mapping[str, np.ndarray],
    *,
    n_prototypes: int,
    max_fit_documents: int,
    max_iter: int,
    random_state: int,
    assignment_batch_size: int = 4096,
) -> CorpusPrototypeGeometry:
    """Fit one query-only control codebook from training corpora only."""
    if not document_embeddings:
        raise ValueError("document_embeddings must not be empty")
    rng = np.random.default_rng(random_state)
    per_corpus = max(
        int(max_fit_documents) // len(document_embeddings), int(n_prototypes)
    )
    samples = []
    for name in sorted(document_embeddings):
        documents = _normalized_rows(document_embeddings[name], str(name))
        size = min(per_corpus, len(documents))
        rows = np.sort(rng.choice(len(documents), size=size, replace=False))
        samples.append(documents[rows])
    fit_documents = np.concatenate(samples)
    sampled = CorpusPrototypeGeometry.fit(
        fit_documents,
        n_prototypes=n_prototypes,
        max_fit_documents=len(fit_documents),
        max_iter=max_iter,
        random_state=random_state,
        assignment_batch_size=assignment_batch_size,
    )
    counts = np.zeros(len(sampled.prototypes), dtype=np.int64)
    for name in sorted(document_embeddings):
        counts += prototype_counts(
            document_embeddings[name],
            sampled.prototypes,
            batch_size=assignment_batch_size,
        )
    return CorpusPrototypeGeometry(sampled.prototypes, counts)


def query_correction_features(
    query_embeddings: np.ndarray,
    method: str,
    local_geometry: CorpusPrototypeGeometry,
    reference_geometry: CorpusPrototypeGeometry,
    projection: np.ndarray,
    *,
    top_m: int,
    temperature: float,
    local_cell_moments: ProjectedCellMoments | None = None,
    reference_cell_moments: ProjectedCellMoments | None = None,
) -> np.ndarray:
    """Build matched q-only, reference, or corpus-local correction features."""
    queries = np.asarray(query_embeddings, dtype=np.float32)
    if method == "q_only":
        return queries
    if method == "reference_geometry":
        geometry = reference_geometry
        cell_moments = reference_cell_moments
    elif method == "corpus_geometry":
        geometry = local_geometry
        cell_moments = local_cell_moments
    else:
        raise ValueError(f"unknown feature method: {method}")
    local = query_local_geometry_features(
        queries,
        geometry,
        projection,
        top_m=min(top_m, len(geometry.prototypes)),
        temperature=temperature,
        cell_moments=cell_moments,
    )
    return np.concatenate([queries, local], axis=1)


__all__ = [
    "CorpusPrototypeGeometry",
    "ProjectedCellMoments",
    "prototype_counts",
    "projected_cell_moments",
    "query_local_geometry_features",
    "fit_reference_geometry",
    "query_correction_features",
]
