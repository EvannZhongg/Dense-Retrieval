"""Shared dataset/cache preparation for correction studies.

Experiment entry points live under ``scripts/`` but their data preparation is
part of the reusable package.  Keeping it here avoids importing one script
from another and makes the split/cache contract testable independently.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from ..datasets import RetrievalDataset, load_beir_dataset
from ..datasets.splits import rows_for_queries
from ..embeddings import MODEL_SPECS, create_embedding_model
from ..embeddings.cache import cached_encode_queries, find_query_cache_path
from .query_correction import load_model_config, prepare_correction_data


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LAMBDAS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]


def _subset_dataset(dataset, name: str, query_ids: set[str]) -> RetrievalDataset:
    queries = [sample for sample in dataset.queries if sample.query_id in query_ids]
    qrels = {sample.query_id: dataset.qrels[sample.query_id] for sample in queries}
    return RetrievalDataset(name, queries, dataset.corpus, qrels)


def _split_holdout(dataset, salt: str):
    ordered = sorted(
        dataset.queries,
        key=lambda sample: hashlib.sha256(f"{salt}:{sample.query_id}".encode()).digest(),
    )
    middle = len(ordered) // 2
    dev_ids = {sample.query_id for sample in ordered[:middle]}
    test_ids = {sample.query_id for sample in ordered[middle:]}
    return (
        _subset_dataset(dataset, f"{dataset.name}_dev", dev_ids),
        _subset_dataset(dataset, f"{dataset.name}_test", test_ids),
    )


def prepare_three_way(
    dataset_name: str,
    model_key: str,
    cache_root: Path,
    *,
    datasets_root: Path | None = None,
    configs_root: Path | None = None,
):
    """Return the common train/dev/test split and cached query embeddings."""
    datasets_root = Path(datasets_root or PROJECT_ROOT / "datasets")
    configs_root = Path(configs_root or PROJECT_ROOT / "configs")
    model_spec = MODEL_SPECS[model_key]
    (
        train_dataset,
        holdout_dataset,
        train_queries,
        holdout_queries,
        documents,
        cache_dir,
        split,
    ) = prepare_correction_data(
        dataset_name,
        model_spec,
        cache_root=cache_root,
        datasets_root=datasets_root,
        configs_root=configs_root,
        split_salt=(
            "arguana-low-rank-v1"
            if dataset_name == "arguana"
            else f"ablation-{dataset_name}-v1"
        ),
        missing_relevant_policy="keep" if dataset_name == "arguana" else "error",
    )
    if dataset_name == "fiqa":
        dev_dataset = load_beir_dataset("fiqa", datasets_root, "dev", False)
        dev_path = find_query_cache_path(cache_dir, "queries_dev.npy")
        if not dev_path.exists():
            model = create_embedding_model(
                load_model_config(configs_root, model_spec)["model"]
            )
            cached_encode_queries(model, dev_dataset.query_texts, dev_path)
        dev_queries = np.load(dev_path, mmap_mode="r")
        test_dataset = holdout_dataset
        test_queries = holdout_queries
        split = "official FiQA train/dev/test qrels"
    else:
        dev_dataset, test_dataset = _split_holdout(
            holdout_dataset,
            "arguana-predictability-v1"
            if dataset_name == "arguana"
            else f"ablation-{dataset_name}-dev-test-v1",
        )
        full_dataset = load_beir_dataset(
            dataset_name,
            datasets_root,
            "test",
            False,
            "keep" if dataset_name == "arguana" else "error",
        )
        full_queries = np.load(cache_dir / "queries.npy", mmap_mode="r")
        dev_queries = full_queries[rows_for_queries(full_dataset, dev_dataset)]
        test_queries = full_queries[rows_for_queries(full_dataset, test_dataset)]
        split = f"{split}; deterministic dev/test split of holdout"
    if len(dev_queries) != len(dev_dataset.queries) or len(test_queries) != len(test_dataset.queries):
        raise ValueError(f"{dataset_name}/{model_key}: query/cache count mismatch")
    return (
        train_dataset,
        dev_dataset,
        test_dataset,
        np.asarray(train_queries),
        np.asarray(dev_queries),
        np.asarray(test_queries),
        documents,
        cache_dir,
        split,
    )


def other_documents(dataset_name: str, model_key: str, cache_root: Path) -> np.ndarray:
    """Load all cached documents from the corpora other than ``dataset_name``."""
    chunks = []
    spec = MODEL_SPECS[model_key]
    for other_name in ("fiqa", "arguana", "scifact", "nfcorpus"):
        if other_name == dataset_name:
            continue
        model_dir = Path(cache_root) / other_name / spec["cache_dir"]
        candidates = sorted(
            path
            for path in model_dir.iterdir()
            if (path / "documents.npy").exists() and (path / "queries.npy").exists()
        )
        if len(candidates) != 1:
            raise RuntimeError(
                f"Expected one cache for {other_name}/{model_key}, got {candidates}"
            )
        chunks.append(np.asarray(np.load(candidates[0] / "documents.npy", mmap_mode="r")))
    if not chunks:
        raise ValueError("at least one external corpus is required")
    return np.concatenate(chunks, axis=0)


__all__ = ["DEFAULT_LAMBDAS", "prepare_three_way", "other_documents"]
