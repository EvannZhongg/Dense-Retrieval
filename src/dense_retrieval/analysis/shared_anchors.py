"""Shared anchor codebooks and corpus occupancy statistics.

An anchor codebook is fitted once from the union of document embeddings for an
embedding model.  Corpora then keep only statistics for the fixed cells.  This
module deliberately has no retrieval or supervision dependencies, so the same
codebook can be reused by regression and ranking-conditioned predictors.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional

import numpy as np

from .prototype_correction import fit_spherical_kmeans


_EPS = 1e-12


def _matrix(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty two-dimensional array")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def _rows(values: np.ndarray, name: str) -> np.ndarray:
    array = _matrix(values, name)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms <= _EPS):
        raise ValueError(f"{name} contains a zero-norm row")
    return array / norms


def _anchors(values: np.ndarray) -> np.ndarray:
    return _rows(values, "anchors")


def _projection(projection: Optional[np.ndarray], dimension: int) -> Optional[np.ndarray]:
    if projection is None:
        return None
    basis = _matrix(projection, "projection")
    if basis.shape[1] == dimension:
        return basis
    if basis.shape[0] == dimension:
        return basis.T
    raise ValueError("projection must have one axis matching embedding dimension")


def fit_global_anchor_codebook(
    document_embeddings: np.ndarray | Mapping[str, np.ndarray],
    n_anchors: int = 256,
    *,
    max_iter: int = 100,
    n_init: int = 1,
    tol: float = 1e-5,
    random_state: Optional[int] = 0,
) -> np.ndarray:
    """Fit one spherical codebook from all corpora for an embedding model.

    A mapping is concatenated in sorted key order to make the result
    independent of dictionary insertion order.  The returned rows are unit
    vectors and are suitable for cosine assignment and query gating.
    """
    if isinstance(document_embeddings, Mapping):
        if not document_embeddings:
            raise ValueError("document_embeddings mapping must not be empty")
        values = [_matrix(document_embeddings[key], str(key)) for key in sorted(document_embeddings)]
        dimensions = {value.shape[1] for value in values}
        if len(dimensions) != 1:
            raise ValueError("all corpus embeddings must have the same dimension")
        documents = np.concatenate(values, axis=0)
    else:
        documents = _matrix(document_embeddings, "document_embeddings")
    return fit_spherical_kmeans(
        documents,
        n_clusters=int(n_anchors),
        max_iter=max_iter,
        n_init=n_init,
        tol=tol,
        random_state=random_state,
    )


# A descriptive alias for callers that use the terminology from the design.
fit_shared_anchor_codebook = fit_global_anchor_codebook


@dataclass
class CorpusAnchorOccupancy:
    """Corpus-side statistics aligned to one fixed global anchor codebook."""

    occupancy: np.ndarray
    counts: np.ndarray
    mean_distance: np.ndarray
    variance_distance: np.ndarray
    local_covariance: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        self.occupancy = np.asarray(self.occupancy, dtype=np.float64)
        self.counts = np.asarray(self.counts, dtype=np.int64)
        self.mean_distance = np.asarray(self.mean_distance, dtype=np.float64)
        self.variance_distance = np.asarray(self.variance_distance, dtype=np.float64)
        if self.occupancy.ndim != 1:
            raise ValueError("occupancy must be one-dimensional")
        k = len(self.occupancy)
        if any(len(value) != k for value in (self.counts, self.mean_distance, self.variance_distance)):
            raise ValueError("occupancy statistics must have one row per anchor")
        if np.any(self.occupancy < 0) or not np.isclose(self.occupancy.sum(), 1.0):
            raise ValueError("occupancy must be non-negative and sum to one")
        if np.any(self.counts < 0) or np.any(self.mean_distance < 0) or np.any(self.variance_distance < 0):
            raise ValueError("occupancy statistics must be non-negative")
        if self.local_covariance is not None:
            self.local_covariance = np.asarray(self.local_covariance, dtype=np.float64)
            if self.local_covariance.ndim != 2 or self.local_covariance.shape[0] != k:
                raise ValueError("local_covariance must have shape (n_anchors, n_features)")

    @property
    def log_occupancy(self) -> np.ndarray:
        """Stable log occupancy used in query-anchor features."""
        return np.log(np.maximum(self.occupancy, _EPS))


def compute_corpus_anchor_occupancy(
    document_embeddings: np.ndarray,
    anchors: np.ndarray,
    *,
    projection: Optional[np.ndarray] = None,
) -> CorpusAnchorOccupancy:
    """Assign documents to nearest anchors and compute per-cell statistics.

    Assignment is hard nearest-anchor assignment under cosine similarity.  The
    distance statistics use ``1 - cosine(document, anchor)``.  When a
    projection is supplied, ``local_covariance`` contains the diagonal
    covariance in that low-dimensional projected space for each cell.
    """
    documents = _rows(document_embeddings, "document_embeddings")
    centers = _anchors(anchors)
    if documents.shape[1] != centers.shape[1]:
        raise ValueError("document_embeddings and anchors must have the same dimension")
    basis = _projection(projection, documents.shape[1])
    similarities = documents @ centers.T
    assignments = np.argmax(similarities, axis=1)
    k = len(centers)
    counts = np.bincount(assignments, minlength=k).astype(np.int64)
    occupancy = counts.astype(np.float64) / float(len(documents))
    distances = np.maximum(1.0 - similarities[np.arange(len(documents)), assignments], 0.0)
    mean_distance = np.zeros(k, dtype=np.float64)
    variance_distance = np.zeros(k, dtype=np.float64)
    local_covariance = None if basis is None else np.zeros((k, basis.shape[0]), dtype=np.float64)
    for index in range(k):
        members = np.flatnonzero(assignments == index)
        if len(members) == 0:
            continue
        cell_distances = distances[members]
        mean_distance[index] = float(np.mean(cell_distances))
        variance_distance[index] = float(np.var(cell_distances))
        if basis is not None and len(members) > 1:
            projected = documents[members] @ basis.T
            local_covariance[index] = np.var(projected, axis=0)
    return CorpusAnchorOccupancy(
        occupancy=occupancy,
        counts=counts,
        mean_distance=mean_distance,
        variance_distance=variance_distance,
        local_covariance=local_covariance,
    )


compute_anchor_occupancy = compute_corpus_anchor_occupancy


def compute_corpus_occupancies(
    document_embeddings: Mapping[str, np.ndarray],
    anchors: np.ndarray,
    *,
    projection: Optional[np.ndarray] = None,
) -> dict[str, CorpusAnchorOccupancy]:
    """Compute aligned occupancy statistics for several corpora."""
    return {
        str(name): compute_corpus_anchor_occupancy(values, anchors, projection=projection)
        for name, values in document_embeddings.items()
    }


@dataclass(frozen=True)
class QueryAnchorFeatures:
    """Sparse Top-M query/corpus features and their fixed anchor indices."""

    values: np.ndarray
    indices: np.ndarray
    similarities: np.ndarray


def select_top_anchors(
    query_embeddings: np.ndarray, anchors: np.ndarray, top_m: int = 16
) -> tuple[np.ndarray, np.ndarray]:
    """Return stable Top-M anchor indices and cosine similarities."""
    queries = _rows(query_embeddings, "query_embeddings")
    centers = _anchors(anchors)
    if queries.shape[1] != centers.shape[1]:
        raise ValueError("query_embeddings and anchors must have the same dimension")
    if not isinstance(top_m, (int, np.integer)) or not 1 <= top_m <= len(centers):
        raise ValueError("top_m must be an integer in [1, number of anchors]")
    similarities = queries @ centers.T
    indices = np.argsort(-similarities, axis=1, kind="mergesort")[:, : int(top_m)]
    return indices, np.take_along_axis(similarities, indices, axis=1)


def build_query_anchor_features(
    query_embeddings: np.ndarray,
    anchors: np.ndarray,
    occupancy: CorpusAnchorOccupancy | np.ndarray,
    *,
    top_m: int = 16,
    projection: Optional[np.ndarray] = None,
    include_local_shape: bool = False,
) -> QueryAnchorFeatures:
    """Construct ``[B.T(anchor-query), q·anchor, log(p_D+eps)]`` per Top-M cell.

    The returned values have shape ``(n_queries, top_m, feature_width)``.  The
    query-only fields are therefore identical for a fixed query and codebook;
    only occupancy and optional local shape fields vary by corpus.
    """
    queries = _rows(query_embeddings, "query_embeddings")
    centers = _anchors(anchors)
    if queries.shape[1] != centers.shape[1]:
        raise ValueError("query_embeddings and anchors must have the same dimension")
    if isinstance(occupancy, CorpusAnchorOccupancy):
        stats = occupancy
    else:
        values = np.asarray(occupancy, dtype=np.float64)
        if values.ndim != 1 or len(values) != len(centers):
            raise ValueError("occupancy must have one value per anchor")
        if not np.all(np.isfinite(values)) or np.any(values < 0) or values.sum() <= _EPS:
            raise ValueError("occupancy must be finite, non-negative, and non-empty")
        values = values / values.sum()
        stats = CorpusAnchorOccupancy(
            values,
            np.rint(values / max(values.sum(), _EPS)).astype(np.int64),
            np.zeros(len(values)),
            np.zeros(len(values)),
        )
    if len(stats.occupancy) != len(centers):
        raise ValueError("occupancy must have one value per anchor")
    basis = _projection(projection, centers.shape[1])
    indices, selected_scores = select_top_anchors(queries, centers, top_m)
    selected_centers = centers[indices]
    relative = selected_centers - queries[:, None, :]
    if basis is None:
        direction = relative
    else:
        direction = relative @ basis.T
    selected_log_p = stats.log_occupancy[indices][..., None]
    values = np.concatenate(
        [direction, selected_scores[..., None], selected_log_p], axis=2
    )
    if include_local_shape:
        shape = np.stack(
            [stats.mean_distance[indices], stats.variance_distance[indices]], axis=2
        )
        if stats.local_covariance is not None:
            shape = np.concatenate([shape, stats.local_covariance[indices]], axis=2)
        values = np.concatenate([values, shape], axis=2)
    return QueryAnchorFeatures(
        values=values.astype(np.float32),
        indices=indices.astype(np.int64),
        similarities=selected_scores.astype(np.float32),
    )


query_anchor_features = build_query_anchor_features


@dataclass
class SharedAnchorCodebook:
    """Reusable global anchors with corpus occupancy and query feature helpers."""

    anchors: np.ndarray
    top_m: int = 16
    projection: Optional[np.ndarray] = None
    epsilon: float = 1e-12

    def __post_init__(self) -> None:
        self.anchors = _anchors(self.anchors)
        self.projection = _projection(self.projection, self.anchors.shape[1])
        if not isinstance(self.top_m, (int, np.integer)) or not 1 <= self.top_m <= len(self.anchors):
            raise ValueError("top_m must be an integer in [1, number of anchors]")
        if self.epsilon <= 0 or not np.isfinite(self.epsilon):
            raise ValueError("epsilon must be finite and positive")

    @classmethod
    def fit(cls, document_embeddings, n_anchors=256, **kwargs):
        top_m = kwargs.pop("top_m", 16)
        projection = kwargs.pop("projection", None)
        anchors = fit_global_anchor_codebook(document_embeddings, n_anchors, **kwargs)
        return cls(anchors, top_m=top_m, projection=projection)

    def occupancy(self, document_embeddings: np.ndarray) -> CorpusAnchorOccupancy:
        return compute_corpus_anchor_occupancy(
            document_embeddings, self.anchors, projection=self.projection
        )

    def query_features(self, query_embeddings, occupancy, *, include_local_shape=False):
        return build_query_anchor_features(
            query_embeddings,
            self.anchors,
            occupancy,
            top_m=self.top_m,
            projection=self.projection,
            include_local_shape=include_local_shape,
        )

    def save(self, path: str | Path, **occupancies: CorpusAnchorOccupancy) -> None:
        """Save anchors and optional named corpus occupancy arrays to one NPZ."""
        payload = {"anchors": self.anchors.astype(np.float32), "top_m": np.asarray([self.top_m])}
        if self.projection is not None:
            payload["projection"] = self.projection.astype(np.float32)
        for name, stats in occupancies.items():
            prefix = f"occupancy/{name}"
            payload[f"{prefix}/occupancy"] = stats.occupancy.astype(np.float32)
            payload[f"{prefix}/counts"] = stats.counts
            payload[f"{prefix}/mean_distance"] = stats.mean_distance.astype(np.float32)
            payload[f"{prefix}/variance_distance"] = stats.variance_distance.astype(np.float32)
            if stats.local_covariance is not None:
                payload[f"{prefix}/local_covariance"] = stats.local_covariance.astype(np.float32)
        np.savez_compressed(path, **payload)

    @classmethod
    def load(cls, path: str | Path) -> "SharedAnchorCodebook":
        with np.load(path) as values:
            projection = values["projection"] if "projection" in values else None
            return cls(values["anchors"], int(values["top_m"][0]), projection=projection)

    @staticmethod
    def load_occupancies(path: str | Path) -> dict[str, CorpusAnchorOccupancy]:
        """Load all named corpus statistics written by :meth:`save`."""
        result = {}
        with np.load(path) as values:
            names = {
                key.split("/", 2)[1]
                for key in values.files
                if key.startswith("occupancy/") and key.count("/") == 2
            }
            for name in names:
                prefix = f"occupancy/{name}"
                covariance_key = f"{prefix}/local_covariance"
                result[name] = CorpusAnchorOccupancy(
                    values[f"{prefix}/occupancy"],
                    values[f"{prefix}/counts"],
                    values[f"{prefix}/mean_distance"],
                    values[f"{prefix}/variance_distance"],
                    values[covariance_key] if covariance_key in values else None,
                )
        return result


__all__ = [
    "CorpusAnchorOccupancy",
    "QueryAnchorFeatures",
    "SharedAnchorCodebook",
    "fit_global_anchor_codebook",
    "fit_shared_anchor_codebook",
    "compute_corpus_anchor_occupancy",
    "compute_anchor_occupancy",
    "compute_corpus_occupancies",
    "select_top_anchors",
    "build_query_anchor_features",
    "query_anchor_features",
]
