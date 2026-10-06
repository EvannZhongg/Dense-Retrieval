"""Deterministic query splits and query-row lookups.

Splits are keyed by a SHA-256 ordering of ``<salt>:<query_id>`` so that they are
reproducible across machines and insensitive to filesystem or dict ordering.  The
salt is supplied by the caller because two studies deliberately use different
splits of the same dataset.
"""
from __future__ import annotations

import hashlib

import numpy as np

from .base import RetrievalDataset


def subset_dataset(dataset, name: str, query_ids) -> RetrievalDataset:
    """Return the queries in ``query_ids``, keeping the shared corpus."""
    ids = set(query_ids)
    queries = [sample for sample in dataset.queries if sample.query_id in ids]
    qrels = {sample.query_id: dataset.qrels[sample.query_id] for sample in queries}
    return RetrievalDataset(name, queries, dataset.corpus, qrels)


def stable_query_split(dataset, *, salt: str, train_fraction: float = 0.7):
    """Split queries deterministically and return a disjoint train/holdout pair."""
    ordered = sorted(
        dataset.queries,
        key=lambda sample: hashlib.sha256(
            f"{salt}:{sample.query_id}".encode()
        ).digest(),
    )
    split_index = int(round(len(ordered) * train_fraction))
    train_ids = {sample.query_id for sample in ordered[:split_index]}
    holdout_ids = {sample.query_id for sample in ordered[split_index:]}
    return (
        subset_dataset(dataset, f"{dataset.name}_train", train_ids),
        subset_dataset(dataset, f"{dataset.name}_holdout", holdout_ids),
    )


def rows_for_queries(full_dataset, subset) -> np.ndarray:
    """Map the queries of ``subset`` to their rows in ``full_dataset``."""
    index = {
        sample.query_id: row for row, sample in enumerate(full_dataset.queries)
    }
    return np.asarray([index[sample.query_id] for sample in subset.queries])
