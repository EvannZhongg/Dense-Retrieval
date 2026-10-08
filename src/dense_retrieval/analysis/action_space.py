"""Correction action spaces for frozen dense retrieval.

The helpers in this module separate two questions that ordinary delta PCA
conflates: how well a basis reconstructs nearest-positive displacements, and
how well it covers directions that change a local ranking boundary.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F

from .ranking_correction import build_rank_records, multi_positive_hard_negative_loss


_EPS = 1e-12


@dataclass(frozen=True)
class RankingOracleSolution:
    """One independently optimized set of privileged query corrections."""

    query_rows: np.ndarray
    coordinates: np.ndarray
    deltas: np.ndarray
    rank_loss: float


def tangent_projection(vectors: np.ndarray, queries: np.ndarray) -> np.ndarray:
    """Project row vectors onto the unit-sphere tangent plane at each query."""
    values = np.asarray(vectors, dtype=np.float64)
    points = np.asarray(queries, dtype=np.float64)
    if values.shape != points.shape or values.ndim != 2:
        raise ValueError("vectors and queries must be same-shaped 2-D arrays")
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    unit_queries = points / np.maximum(norms, _EPS)
    radial = np.sum(values * unit_queries, axis=1, keepdims=True)
    return values - radial * unit_queries


def fit_svd_subspace(samples: np.ndarray, max_rank: int) -> np.ndarray:
    """Fit a zero-origin nested subspace to row samples by SVD."""
    values = np.asarray(samples, dtype=np.float64)
    if values.ndim != 2 or len(values) == 0:
        raise ValueError("samples must be a non-empty 2-D array")
    if not 1 <= int(max_rank) <= min(values.shape):
        raise ValueError("max_rank must be between 1 and min(samples.shape)")
    _, _, components = np.linalg.svd(values, full_matrices=False)
    return components[: int(max_rank)]


def random_subspace(dimension: int, max_rank: int, random_state: int) -> np.ndarray:
    """Return a reproducible nested random orthonormal row basis."""
    if not 1 <= int(max_rank) <= int(dimension):
        raise ValueError("max_rank must be between 1 and dimension")
    rng = np.random.default_rng(random_state)
    values = rng.standard_normal((int(dimension), int(max_rank)))
    basis, _ = np.linalg.qr(values, mode="reduced")
    return basis.T


def project_to_subspace(
    vectors: np.ndarray, components: np.ndarray, rank: int
) -> np.ndarray:
    """Orthogonally project row vectors into the leading nested subspace."""
    values = np.asarray(vectors, dtype=np.float64)
    basis = np.asarray(components, dtype=np.float64)[: int(rank)]
    if values.ndim != 2 or basis.ndim != 2 or values.shape[1] != basis.shape[1]:
        raise ValueError("vectors and components must have the same feature width")
    if not 1 <= int(rank) <= len(components):
        raise ValueError("rank must be between 1 and the number of components")
    return (values @ basis.T) @ basis


def projection_statistics(
    vectors: np.ndarray, projected: np.ndarray
) -> dict[str, float]:
    """Summarize reconstruction separately from downstream ranking utility."""
    values = np.asarray(vectors, dtype=np.float64)
    estimates = np.asarray(projected, dtype=np.float64)
    if values.shape != estimates.shape or values.ndim != 2:
        raise ValueError("vectors and projected must be same-shaped 2-D arrays")
    residual = values - estimates
    total_energy = float(np.sum(values * values))
    residual_energy = float(np.sum(residual * residual))
    dot = np.sum(values * estimates, axis=1)
    cosine = dot / np.maximum(
        np.linalg.norm(values, axis=1) * np.linalg.norm(estimates, axis=1), _EPS
    )
    return {
        "energy_retention": 1.0 - residual_energy / max(total_energy, _EPS),
        "mean_squared_error": float(np.mean(residual * residual)),
        "mean_cosine": float(np.mean(cosine)),
    }


def ranking_boundary_directions(
    dataset,
    query_embeddings: np.ndarray,
    document_embeddings: np.ndarray,
    *,
    hard_negatives: int = 32,
    temperature: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Return tangent first-order directions that improve hard-negative margins.

    For each labeled training query, the positive is its currently highest
    scoring relevant document. Hard negatives are weighted by a softmax of
    their baseline scores. The resulting ``d_positive - E[d_negative]`` is the
    descent direction of a smooth pairwise ranking loss, projected onto the
    query sphere's tangent plane because corrected queries are renormalized.
    """
    if hard_negatives < 1:
        raise ValueError("hard_negatives must be positive")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    queries = np.asarray(query_embeddings, dtype=np.float32)
    documents = np.asarray(document_embeddings, dtype=np.float32)
    records = build_rank_records(dataset, queries, documents, hard_negatives)
    rows = records["query_rows"]
    directions = []
    for record_row, query_row in enumerate(rows):
        query = queries[query_row]
        positive_indices = records["positive_indices"][record_row]
        positive_indices = positive_indices[positive_indices >= 0]
        negative_indices = records["negative_indices"][record_row]
        negative_indices = negative_indices[negative_indices >= 0]
        positive_docs = documents[positive_indices]
        negative_docs = documents[negative_indices]
        positive = positive_docs[np.argmax(positive_docs @ query)]
        logits = (negative_docs @ query) / float(temperature)
        logits -= np.max(logits)
        weights = np.exp(logits)
        weights /= np.sum(weights)
        directions.append(positive - weights @ negative_docs)
    values = np.asarray(directions, dtype=np.float64)
    return rows, tangent_projection(values, queries[rows])


def optimize_ranking_oracle(
    query_embeddings: np.ndarray,
    document_embeddings: np.ndarray,
    rank_records: dict[str, np.ndarray],
    components: np.ndarray | None,
    *,
    steps: int = 100,
    learning_rate: float = 0.05,
    penalty: float = 1e-3,
    temperature: float = 0.05,
    device: str = "auto",
) -> tuple[np.ndarray, np.ndarray, float]:
    """Optimize privileged per-query corrections within one fixed subspace.

    Passing ``components=None`` uses the full ambient space. Hard-negative
    indices must be built once at the frozen baseline and supplied by the
    caller so every candidate subspace sees the same ranking boundary.
    """
    solution = optimize_ranking_oracle_solution(
        query_embeddings,
        document_embeddings,
        rank_records,
        components,
        steps=steps,
        learning_rate=learning_rate,
        penalty=penalty,
        temperature=temperature,
        device=device,
    )
    return solution.query_rows, solution.deltas, solution.rank_loss


def optimize_ranking_oracle_solution(
    query_embeddings: np.ndarray,
    document_embeddings: np.ndarray,
    rank_records: dict[str, np.ndarray],
    components: np.ndarray | None,
    *,
    steps: int = 100,
    learning_rate: float = 0.05,
    penalty: float = 1e-3,
    temperature: float = 0.05,
    device: str = "auto",
    initial_coordinates: np.ndarray | None = None,
    optimizer_seed: int = 0,
) -> RankingOracleSolution:
    """Optimize and expose both coordinates and ambient displacements.

    ``optimizer_seed`` is explicit even though Adam is deterministic here. It
    provides a useful control: changing the seed alone should not change the
    solution unless the execution backend introduces nondeterminism.
    """
    if steps < 1 or learning_rate <= 0 or penalty < 0:
        raise ValueError("steps and learning_rate must be positive; penalty non-negative")
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be one of: auto, cpu, cuda")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    resolved = torch.device(
        "cuda" if device == "auto" and torch.cuda.is_available() else "cpu" if device == "auto" else device
    )
    rows = np.asarray(rank_records["query_rows"], dtype=np.int64)
    queries = np.asarray(query_embeddings, dtype=np.float32)[rows]
    documents = np.asarray(document_embeddings, dtype=np.float32)
    positive_indices = np.asarray(rank_records["positive_indices"], dtype=np.int64)
    negative_indices = np.asarray(rank_records["negative_indices"], dtype=np.int64)
    positive_mask = positive_indices >= 0
    negative_mask = negative_indices >= 0
    positive_docs = documents[np.maximum(positive_indices, 0)]
    negative_docs = documents[np.maximum(negative_indices, 0)]

    q = torch.from_numpy(queries).to(resolved)
    positives = torch.from_numpy(positive_docs).to(resolved)
    negatives = torch.from_numpy(negative_docs).to(resolved)
    positive_mask_tensor = torch.from_numpy(positive_mask).to(resolved)
    negative_mask_tensor = torch.from_numpy(negative_mask).to(resolved)
    if components is None:
        basis = None
        parameter_width = queries.shape[1]
    else:
        component_values = np.asarray(components, dtype=np.float32)
        if component_values.ndim != 2 or component_values.shape[1] != queries.shape[1]:
            raise ValueError("components must be rank by embedding dimension")
        basis = torch.from_numpy(component_values).to(resolved)
        parameter_width = len(component_values)
    expected_shape = (len(rows), parameter_width)
    if initial_coordinates is None:
        initial_values = np.zeros(expected_shape, dtype=np.float32)
    else:
        initial_values = np.asarray(initial_coordinates, dtype=np.float32)
        if initial_values.shape != expected_shape:
            raise ValueError(
                f"initial_coordinates must have shape {expected_shape}, got "
                f"{initial_values.shape}"
            )
    torch.manual_seed(int(optimizer_seed))
    if resolved.type == "cuda":
        torch.cuda.manual_seed_all(int(optimizer_seed))
    coordinates = torch.tensor(
        initial_values, dtype=torch.float32, device=resolved, requires_grad=True
    )
    optimizer = torch.optim.Adam([coordinates], lr=float(learning_rate))
    for _ in range(int(steps)):
        delta = coordinates if basis is None else coordinates @ basis
        corrected = F.normalize(q + delta, dim=1)
        positive_scores = torch.einsum("bd,bpd->bp", corrected, positives)
        negative_scores = torch.einsum("bd,bnd->bn", corrected, negatives)
        rank_loss = multi_positive_hard_negative_loss(
            positive_scores,
            negative_scores,
            positive_mask_tensor,
            negative_mask_tensor,
            temperature=temperature,
        )
        loss = rank_loss + float(penalty) * torch.mean(torch.sum(delta * delta, dim=1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        delta = coordinates if basis is None else coordinates @ basis
        corrected = F.normalize(q + delta, dim=1)
        final_loss = multi_positive_hard_negative_loss(
            torch.einsum("bd,bpd->bp", corrected, positives),
            torch.einsum("bd,bnd->bn", corrected, negatives),
            positive_mask_tensor,
            negative_mask_tensor,
            temperature=temperature,
        )
    return RankingOracleSolution(
        query_rows=rows,
        coordinates=coordinates.detach().cpu().numpy(),
        deltas=delta.detach().cpu().numpy(),
        rank_loss=float(final_loss.cpu()),
    )


__all__ = [
    "fit_svd_subspace",
    "optimize_ranking_oracle",
    "optimize_ranking_oracle_solution",
    "project_to_subspace",
    "projection_statistics",
    "random_subspace",
    "ranking_boundary_directions",
    "tangent_projection",
    "RankingOracleSolution",
]
