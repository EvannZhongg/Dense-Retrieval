"""Discrete correction actions for utility-aligned query calibration.

The action set is learned only from labeled training corpora.  At inference a
policy chooses one action, including exact abstention, before the single final
retrieval.  This avoids treating one privileged coordinate vector as the only
valid target when several corrections may have comparable ranking utility.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .prototype_correction import fit_spherical_kmeans


_EPS = 1e-12


def fit_candidate_actions(
    coordinate_sets: Sequence[np.ndarray],
    *,
    n_directions: int,
    magnitudes: Sequence[float],
    max_samples_per_corpus: int = 1000,
    random_state: int = 0,
) -> np.ndarray:
    """Fit corpus-balanced direction prototypes and prepend a zero action."""
    if not coordinate_sets:
        raise ValueError("coordinate_sets must not be empty")
    if n_directions < 1 or max_samples_per_corpus < 1:
        raise ValueError("n_directions and max_samples_per_corpus must be positive")
    steps = np.asarray(magnitudes, dtype=np.float64)
    if steps.ndim != 1 or not len(steps) or np.any(steps <= 0):
        raise ValueError("magnitudes must be a non-empty sequence of positive values")

    rng = np.random.default_rng(random_state)
    sampled = []
    width = None
    for values in coordinate_sets:
        coordinates = np.asarray(values, dtype=np.float64)
        if coordinates.ndim != 2 or not len(coordinates):
            raise ValueError("every coordinate set must be a non-empty 2-D array")
        if width is None:
            width = coordinates.shape[1]
        elif coordinates.shape[1] != width:
            raise ValueError("all coordinate sets must have equal width")
        norms = np.linalg.norm(coordinates, axis=1)
        usable = np.flatnonzero(norms > _EPS)
        if not len(usable):
            continue
        size = min(int(max_samples_per_corpus), len(usable))
        rows = np.sort(rng.choice(usable, size=size, replace=False))
        sampled.append(coordinates[rows] / norms[rows, None])
    if not sampled:
        raise ValueError("coordinate sets contain no non-zero actions")

    directions = fit_spherical_kmeans(
        np.concatenate(sampled),
        n_clusters=min(int(n_directions), sum(len(values) for values in sampled)),
        max_iter=30,
        random_state=random_state,
    )
    actions = [np.zeros((1, directions.shape[1]), dtype=np.float64)]
    actions.extend(float(step) * directions for step in steps)
    return np.concatenate(actions).astype(np.float32)


def action_query_features(
    query_embeddings: np.ndarray,
    action_coordinates: np.ndarray,
    components: np.ndarray,
) -> np.ndarray:
    """Return compact query/action interactions for every query-action pair.

    Output order is query-major: all actions for query zero, followed by all
    actions for query one.  The same order is used by :func:`reshape_utilities`.
    """
    queries = np.asarray(query_embeddings, dtype=np.float64)
    actions = np.asarray(action_coordinates, dtype=np.float64)
    basis = np.asarray(components, dtype=np.float64)
    if queries.ndim != 2 or actions.ndim != 2 or basis.ndim != 2:
        raise ValueError("queries, actions, and components must be 2-D arrays")
    if queries.shape[1] != basis.shape[1] or actions.shape[1] != basis.shape[0]:
        raise ValueError("action/query dimensions do not match components")

    query_coordinates = queries @ basis.T
    q = np.repeat(query_coordinates[:, None, :], len(actions), axis=1)
    a = np.broadcast_to(actions[None, :, :], q.shape)
    corrected = q + a
    dot = np.sum(q * a, axis=2, keepdims=True)
    magnitude = np.linalg.norm(a, axis=2, keepdims=True)
    return np.concatenate([q, a, q * a, corrected, dot, magnitude], axis=2).reshape(
        len(queries) * len(actions), -1
    ).astype(np.float32)


def corrected_query_grid(
    query_embeddings: np.ndarray,
    action_coordinates: np.ndarray,
    components: np.ndarray,
) -> np.ndarray:
    """Materialize normalized corrected queries in query-major order."""
    queries = np.asarray(query_embeddings, dtype=np.float64)
    actions = np.asarray(action_coordinates, dtype=np.float64)
    basis = np.asarray(components, dtype=np.float64)
    if queries.ndim != 2 or actions.ndim != 2 or basis.ndim != 2:
        raise ValueError("queries, actions, and components must be 2-D arrays")
    deltas = actions @ basis
    corrected = queries[:, None, :] + deltas[None, :, :]
    corrected /= np.maximum(np.linalg.norm(corrected, axis=2, keepdims=True), _EPS)
    return corrected.reshape(len(queries) * len(actions), queries.shape[1]).astype(
        np.float32
    )


def reshape_utilities(values: np.ndarray, query_count: int, action_count: int) -> np.ndarray:
    """Validate and reshape a query-major flat prediction array."""
    array = np.asarray(values)
    if array.size != int(query_count) * int(action_count):
        raise ValueError("utility count does not match query/action dimensions")
    return array.reshape(int(query_count), int(action_count))


__all__ = [
    "action_query_features",
    "corrected_query_grid",
    "fit_candidate_actions",
    "reshape_utilities",
]
