"""Query-side correction primitives shared by the correction study scripts.

The correction target of one query is the difference between a positive document
embedding and the query embedding.  :func:`build_oracle_deltas` selects the
closest available positive per query, :func:`fit_pca_subspace` and
:func:`project_deltas` fit and apply a low-rank basis over those targets, and
:func:`apply_correction` moves a normalized query inside that basis.  Hosted
models are only rebuilt here when a cached query encoding is missing, so the
studies stay reproducible from the cache.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from ..datasets import load_beir_dataset
from ..datasets.splits import rows_for_queries, stable_query_split
from ..embeddings import create_embedding_model
from ..embeddings.cache import (
    cached_encode_queries,
    find_cache_dir,
    find_query_cache_path,
)
from ..evaluation.oracle import closest_positive_index


def load_model_config(configs_root, model_spec):
    """Read the config that can rebuild ``model_spec``'s hosted embedding model."""
    path = Path(configs_root) / model_spec["config"]
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def build_oracle_deltas(dataset, query_embeddings, document_embeddings):
    """Return ``(query_rows, deltas, missing_query_ids)`` for the closest positives."""
    document_index = {
        doc_id: index for index, doc_id in enumerate(dataset.corpus)
    }
    query_rows = []
    deltas = []
    missing_query_ids = []
    for query_row, sample in enumerate(dataset.queries):
        positive_index = closest_positive_index(
            sample.query_id,
            dataset.qrels,
            query_embeddings[query_row],
            document_embeddings,
            document_index,
        )
        if positive_index is None:
            missing_query_ids.append(sample.query_id)
            continue
        query = np.asarray(query_embeddings[query_row], dtype=np.float64)
        positive = np.asarray(document_embeddings[positive_index], dtype=np.float64)
        query_rows.append(query_row)
        deltas.append(positive - query)
    return (
        np.asarray(query_rows, dtype=np.int64),
        np.asarray(deltas, dtype=np.float64),
        missing_query_ids,
    )


def select_oracle_positives(dataset, query_embeddings, document_embeddings):
    """Return dense positive embeddings plus a mask of queries that have one."""
    document_index = {
        doc_id: index for index, doc_id in enumerate(dataset.corpus)
    }
    positives = np.zeros_like(query_embeddings)
    available = np.zeros(len(query_embeddings), dtype=bool)
    for query_index, sample in enumerate(dataset.queries):
        positive_index = closest_positive_index(
            sample.query_id,
            dataset.qrels,
            query_embeddings[query_index],
            document_embeddings,
            document_index,
        )
        if positive_index is not None:
            positives[query_index] = document_embeddings[positive_index]
            available[query_index] = True
    return positives, available


def fit_pca_subspace(train_deltas, max_rank):
    """Fit the mean and leading right singular vectors of the training deltas."""
    mean = train_deltas.mean(axis=0)
    centered = train_deltas - mean
    _, _, components = np.linalg.svd(centered, full_matrices=False)
    return mean, components[:max_rank]


def fit_zero_origin_pca_subspace(train_deltas, max_rank, sample_weights=None):
    """Fit a low-rank subspace whose zero coordinate is exactly zero correction.

    The usual centered PCA stores a non-zero mean correction.  That is useful for
    reconstruction, but it prevents a predictor from opting out for an individual
    query.  This variant fits the basis through the origin; a zero coordinate
    vector therefore reconstructs to an exact zero delta.
    """
    deltas = np.asarray(train_deltas, dtype=np.float64)
    if deltas.ndim != 2 or len(deltas) == 0:
        raise ValueError("train_deltas must be a non-empty 2-D array")
    if sample_weights is None:
        weighted = deltas
    else:
        weights = np.asarray(sample_weights, dtype=np.float64)
        if weights.shape != (len(deltas),) or np.any(weights < 0) or not np.any(weights > 0):
            raise ValueError("sample_weights must be a non-negative vector")
        weighted = deltas * np.sqrt(weights[:, None])
    _, _, components = np.linalg.svd(weighted, full_matrices=False)
    return np.zeros(deltas.shape[1], dtype=np.float64), components[:max_rank]


def project_deltas(deltas, mean, components, rank):
    """Project deltas onto the leading ``rank`` basis vectors, with the mean."""
    basis = components[:rank]
    centered = deltas - mean
    return mean + (centered @ basis.T) @ basis


def apply_correction(query_embeddings, query_rows, deltas, lambda_):
    """Move the selected queries by ``lambda_ * deltas`` and renormalize."""
    corrected = np.asarray(query_embeddings, dtype=np.float64).copy()
    corrected[query_rows] += float(lambda_) * deltas
    corrected /= np.maximum(np.linalg.norm(corrected, axis=1, keepdims=True), 1e-12)
    return corrected.astype(np.float32)


def corrected_queries(queries, deltas, lambda_):
    """Apply a scaled correction to every query and renormalize rows."""
    corrected = np.asarray(queries, dtype=np.float64) + float(lambda_) * np.asarray(
        deltas, dtype=np.float64
    )
    corrected /= np.maximum(np.linalg.norm(corrected, axis=1, keepdims=True), 1e-12)
    return corrected.astype(np.float32)


def apply_oracle_correction(query_embeddings, positives, available, lambda_):
    """Interpolate the available queries toward their positive and renormalize."""
    corrected = np.asarray(query_embeddings, dtype=np.float32).copy()
    corrected[available] = (
        corrected[available] * (1.0 - lambda_)
        + positives[available] * lambda_
    )
    norms = np.linalg.norm(corrected, axis=1, keepdims=True)
    return corrected / np.maximum(norms, 1e-12)


def gain_retention(metric_value, baseline, oracle):
    """Fraction of the baseline-to-oracle gain that ``metric_value`` retains."""
    denominator = oracle - baseline
    if abs(denominator) <= 1e-12:
        return np.nan
    return (metric_value - baseline) / denominator


def prepare_correction_data(
    dataset_name: str,
    model_spec,
    *,
    cache_root,
    datasets_root,
    configs_root,
    split_salt: str | None = None,
    train_fraction: float = 0.7,
    missing_relevant_policy: str = "error",
    encode_missing_train_queries: bool = True,
):
    """Return disjoint train/holdout queries with their cached embeddings.

    FiQA uses the official train/test qrels.  Every other dataset uses the
    official test split and a deterministic SHA-256 split of its queries, where
    ``split_salt`` distinguishes studies that must not share a split.  Callers
    fit their correction parameters on the train queries only; the held-out
    queries are for evaluation.
    """
    cache_dir = find_cache_dir(cache_root, dataset_name, model_spec["cache_dir"])
    documents = np.load(cache_dir / "documents.npy", mmap_mode="r")

    if dataset_name == "fiqa":
        train_dataset = load_beir_dataset("fiqa", datasets_root, "train", False)
        holdout_dataset = load_beir_dataset("fiqa", datasets_root, "test", False)
        train_path = find_query_cache_path(cache_dir, "queries_train.npy")
        if train_path.exists():
            train_queries = np.load(train_path, mmap_mode="r")
        elif encode_missing_train_queries:
            model = create_embedding_model(
                load_model_config(configs_root, model_spec)["model"]
            )
            train_queries = cached_encode_queries(
                model, train_dataset.query_texts, train_path
            )
        else:
            raise FileNotFoundError(f"Missing cached FiQA train queries: {train_path}")
        holdout_queries = np.load(cache_dir / "queries.npy", mmap_mode="r")
        split_method = "official FiQA train/test qrels"
    else:
        full_dataset = load_beir_dataset(
            dataset_name, datasets_root, "test", False, missing_relevant_policy
        )
        train_dataset, holdout_dataset = stable_query_split(
            full_dataset,
            salt=split_salt or f"{dataset_name}-correction-v1",
            train_fraction=train_fraction,
        )
        full_queries = np.load(cache_dir / "queries.npy", mmap_mode="r")
        train_queries = full_queries[rows_for_queries(full_dataset, train_dataset)]
        holdout_queries = full_queries[rows_for_queries(full_dataset, holdout_dataset)]
        train_percent = round(train_fraction * 100)
        split_method = (
            f"stable SHA-256 {train_percent}/{100 - train_percent} split of "
            f"{dataset_name} test queries"
        )

    if len(train_queries) != len(train_dataset.queries):
        raise ValueError(f"{dataset_name}: train query/cache count mismatch")
    if len(holdout_queries) != len(holdout_dataset.queries):
        raise ValueError(f"{dataset_name}: holdout query/cache count mismatch")

    return (
        train_dataset,
        holdout_dataset,
        np.asarray(train_queries),
        np.asarray(holdout_queries),
        documents,
        cache_dir,
        split_method,
    )
