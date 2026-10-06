"""Prototype Correction Field for frozen dense-retrieval embeddings.

The field has two separate pieces of state:

* document-only spherical K-Means prototypes, and
* a linear map from sparse prototype weights to correction coordinates.

The fitting target is deliberately supplied by the caller.  In the low-rank
oracle setup this target is ``(delta - mean) @ components.T`` where
``delta = d_positive - q``; qrels are therefore only needed while constructing
the training targets and never while building the document prototypes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


_EPS = 1e-12


def _as_matrix(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must be non-empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def _normalize_rows(values: np.ndarray, name: str) -> np.ndarray:
    array = _as_matrix(values, name)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms <= _EPS):
        raise ValueError(f"{name} contains a zero-norm row")
    return array / norms


def fit_spherical_kmeans(
    document_embeddings: np.ndarray,
    n_clusters: int = 64,
    *,
    max_iter: int = 100,
    n_init: int = 1,
    tol: float = 1e-5,
    random_state: Optional[int] = 0,
) -> np.ndarray:
    """Fit cosine/spherical K-Means and return ``(n_clusters, dimension)``.

    Rows are normalized before fitting and every updated centroid is
    renormalized.  Initialization uses cosine-distance k-means++ and empty
    clusters are deterministically reseeded with the least represented point.
    ``n_init`` selects the run with the largest cosine objective.
    """
    documents = _normalize_rows(document_embeddings, "document_embeddings")
    n_documents, dimension = documents.shape
    if not isinstance(n_clusters, (int, np.integer)) or not 1 <= n_clusters <= n_documents:
        raise ValueError("n_clusters must be an integer in [1, number of documents]")
    if max_iter < 1 or n_init < 1 or tol < 0:
        raise ValueError("max_iter and n_init must be positive and tol non-negative")

    master = np.random.default_rng(random_state)
    best_centers = None
    best_score = -np.inf
    for _ in range(int(n_init)):
        rng = np.random.default_rng(master.integers(0, np.iinfo(np.int64).max))
        centers = _kmeans_plus_plus_init(documents, int(n_clusters), rng)
        previous_labels = None
        for _iteration in range(int(max_iter)):
            similarities = documents @ centers.T
            labels = np.argmax(similarities, axis=1)
            updated = np.zeros_like(centers)
            counts = np.bincount(labels, minlength=n_clusters)
            for cluster in range(n_clusters):
                members = documents[labels == cluster]
                if len(members):
                    updated[cluster] = members.sum(axis=0)
                    updated[cluster] /= max(np.linalg.norm(updated[cluster]), _EPS)
            # Re-seed empty clusters using points furthest from their assigned
            # center. This keeps all K prototypes usable for sparse gating.
            empty = np.flatnonzero(counts == 0)
            if len(empty):
                nearest = similarities[np.arange(n_documents), labels]
                order = np.argsort(nearest, kind="mergesort")
                used = set()
                cursor = 0
                for cluster in empty:
                    while cursor < len(order) - 1 and int(order[cursor]) in used:
                        cursor += 1
                    point = int(order[cursor])
                    used.add(point)
                    updated[cluster] = documents[point]
                    cursor += 1
            shift = float(np.max(1.0 - np.sum(centers * updated, axis=1)))
            centers = updated
            if previous_labels is not None and np.array_equal(labels, previous_labels):
                break
            if shift <= tol:
                break
            previous_labels = labels
        score = float(np.sum(np.max(documents @ centers.T, axis=1)))
        if score > best_score:
            best_score = score
            best_centers = centers.copy()
    # best_centers is always set because n_init >= 1.
    return np.asarray(best_centers, dtype=np.float64)


def _kmeans_plus_plus_init(
    samples: np.ndarray, n_clusters: int, rng: np.random.Generator
) -> np.ndarray:
    centers = np.empty((n_clusters, samples.shape[1]), dtype=np.float64)
    first = int(rng.integers(0, len(samples)))
    centers[0] = samples[first]
    closest_distance = 1.0 - samples @ centers[0]
    for index in range(1, n_clusters):
        probabilities = np.maximum(closest_distance, 0.0)
        total = float(probabilities.sum())
        if total <= _EPS:
            candidates = np.flatnonzero(
                np.min(np.abs(samples[:, None, :] - centers[None, :index, :]), axis=(1, 2))
                > _EPS
            )
            chosen = int(candidates[0] if len(candidates) else rng.integers(0, len(samples)))
        else:
            chosen = int(rng.choice(len(samples), p=probabilities / total))
        centers[index] = samples[chosen]
        closest_distance = np.minimum(closest_distance, 1.0 - samples @ centers[index])
    return centers


def prototype_weights(
    queries: np.ndarray,
    prototypes: np.ndarray,
    *,
    temperature: float = 0.07,
    top_m: int = 8,
) -> np.ndarray:
    """Compute Top-M sparse prototype weights for each query."""
    query_rows = _normalize_rows(queries, "queries")
    prototype_rows = _normalize_rows(prototypes, "prototypes")
    if query_rows.shape[1] != prototype_rows.shape[1]:
        raise ValueError("queries and prototypes must have the same dimension")
    if temperature <= 0 or not np.isfinite(temperature):
        raise ValueError("temperature must be finite and positive")
    if not isinstance(top_m, (int, np.integer)) or not 1 <= top_m <= len(prototype_rows):
        raise ValueError("top_m must be an integer in [1, number of prototypes]")

    similarities = query_rows @ prototype_rows.T
    # Stable sorting gives deterministic tie handling and keeps the support
    # exactly Top-M, including when similarities are equal.
    top_indices = np.argsort(-similarities, axis=1, kind="mergesort")[:, : int(top_m)]
    selected = np.take_along_axis(similarities, top_indices, axis=1) / float(temperature)
    selected -= selected.max(axis=1, keepdims=True)
    selected = np.exp(selected)
    selected /= np.maximum(selected.sum(axis=1, keepdims=True), _EPS)
    weights = np.zeros_like(similarities, dtype=np.float64)
    rows = np.arange(len(query_rows))[:, None]
    weights[rows, top_indices] = selected
    return weights


def fit_prototype_values(
    weights: np.ndarray, targets: np.ndarray, *, ridge: float = 0.0
) -> np.ndarray:
    """Fit ``V`` in ``targets ~= weights @ V``.

    ``ridge`` is an L2 penalty on prototype values and is useful when sparse
    supports leave some prototypes weakly observed. No intercept is fitted,
    matching the requested convex combination form.
    """
    design = _as_matrix(weights, "weights")
    target = _as_matrix(targets, "targets")
    if design.shape[0] != target.shape[0]:
        raise ValueError("weights and targets must have the same number of rows")
    if ridge < 0 or not np.isfinite(ridge):
        raise ValueError("ridge must be finite and non-negative")
    if ridge == 0:
        values, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
        return values
    gram = design.T @ design
    gram.flat[:: len(gram) + 1] += float(ridge)
    return np.linalg.solve(gram, design.T @ target)


def predict_prototype_values(weights: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Predict correction coordinates from sparse weights and fitted values."""
    design = _as_matrix(weights, "weights")
    prototypes_values = _as_matrix(values, "values")
    if design.shape[1] != prototypes_values.shape[0]:
        raise ValueError("weights columns must match values rows")
    return design @ prototypes_values


def project_correction_targets(
    deltas: np.ndarray, mean: np.ndarray, components: np.ndarray
) -> np.ndarray:
    """Convert full correction deltas to low-rank coordinates."""
    delta_rows = _as_matrix(deltas, "deltas")
    center = np.asarray(mean, dtype=np.float64)
    basis = _as_matrix(components, "components")
    if center.ndim != 1 or center.shape[0] != delta_rows.shape[1]:
        raise ValueError("mean must have one entry per delta dimension")
    if basis.shape[1] != delta_rows.shape[1]:
        raise ValueError("components width must match delta dimension")
    return (delta_rows - center) @ basis.T


@dataclass
class PrototypeCorrectionField:
    """Frozen prototypes plus their learned correction coordinates."""

    prototypes: np.ndarray
    values: np.ndarray
    temperature: float = 0.07
    top_m: int = 8
    mean: Optional[np.ndarray] = None
    components: Optional[np.ndarray] = None

    @classmethod
    def fit(
        cls,
        document_embeddings: np.ndarray,
        query_embeddings: np.ndarray,
        target_coordinates: np.ndarray,
        *,
        n_prototypes: int = 64,
        top_m: int = 8,
        temperature: float = 0.07,
        ridge: float = 0.0,
        max_iter: int = 100,
        n_init: int = 1,
        random_state: Optional[int] = 0,
        mean: Optional[np.ndarray] = None,
        components: Optional[np.ndarray] = None,
    ) -> "PrototypeCorrectionField":
        prototypes = fit_spherical_kmeans(
            document_embeddings,
            n_clusters=n_prototypes,
            max_iter=max_iter,
            n_init=n_init,
            random_state=random_state,
        )
        weights = prototype_weights(
            query_embeddings, prototypes, temperature=temperature, top_m=top_m
        )
        values = fit_prototype_values(weights, target_coordinates, ridge=ridge)
        return cls(
            prototypes=prototypes,
            values=values,
            temperature=float(temperature),
            top_m=int(top_m),
            mean=None if mean is None else np.asarray(mean, dtype=np.float64),
            components=None
            if components is None
            else np.asarray(components, dtype=np.float64),
        )

    def weights(self, query_embeddings: np.ndarray) -> np.ndarray:
        return prototype_weights(
            query_embeddings,
            self.prototypes,
            temperature=self.temperature,
            top_m=self.top_m,
        )

    def predict_coordinates(self, query_embeddings: np.ndarray) -> np.ndarray:
        return predict_prototype_values(self.weights(query_embeddings), self.values)

    def predict_deltas(self, query_embeddings: np.ndarray) -> np.ndarray:
        coordinates = self.predict_coordinates(query_embeddings)
        if self.mean is None or self.components is None:
            return coordinates
        return np.asarray(self.mean) + coordinates @ np.asarray(self.components)

    def apply(
        self, query_embeddings: np.ndarray, *, lambda_: float = 1.0
    ) -> np.ndarray:
        """Apply predicted deltas on the query side and renormalize."""
        queries = _normalize_rows(query_embeddings, "query_embeddings")
        deltas = self.predict_deltas(queries)
        if deltas.shape[1] != queries.shape[1]:
            raise ValueError(
                "predicted correction dimension must match query dimension; "
                "provide mean and components when values are low-rank coordinates"
            )
        corrected = queries + float(lambda_) * deltas
        norms = np.linalg.norm(corrected, axis=1, keepdims=True)
        return corrected / np.maximum(norms, _EPS)


# Short aliases make the mathematical API easy to discover and preserve a
# conventional verb-first spelling for callers that prefer it.
compute_prototype_weights = prototype_weights
compute_sparse_prototype_weights = prototype_weights
spherical_kmeans = fit_spherical_kmeans


def fit_prototype_correction_field(
    document_embeddings: np.ndarray,
    query_embeddings: np.ndarray,
    target_coordinates: np.ndarray,
    **kwargs,
) -> PrototypeCorrectionField:
    """Fit a complete field from frozen documents, queries, and targets."""
    return PrototypeCorrectionField.fit(
        document_embeddings, query_embeddings, target_coordinates, **kwargs
    )
