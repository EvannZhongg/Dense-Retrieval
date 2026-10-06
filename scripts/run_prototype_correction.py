"""Train and evaluate a leakage-free Prototype Correction Field.

Document prototypes are fitted once from frozen corpus embeddings.  Only the
training queries' oracle correction coordinates are used to learn prototype
values; test qrels are used solely for evaluation by the existing oracle
pipeline.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

LOW_RANK_PATH = ROOT / "scripts" / "run_low_rank_oracle.py"
SPEC = importlib.util.spec_from_file_location("run_low_rank_oracle", LOW_RANK_PATH)
low_rank = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(low_rank)

from dense_retrieval.analysis.prototype_correction import (  # noqa: E402
    PrototypeCorrectionField,
    fit_spherical_kmeans,
    fit_prototype_values,
    predict_prototype_values,
    prototype_weights,
    project_correction_targets,
)
from dense_retrieval.datasets import RetrievalDataset, load_beir_dataset  # noqa: E402


def fit_field(
    document_embeddings: np.ndarray,
    query_embeddings: np.ndarray,
    target_coordinates: np.ndarray,
    *,
    n_prototypes: int = 64,
    top_m: int = 8,
    temperature: float = 0.07,
    ridge: float = 1e-3,
    random_state: int = 0,
    mean: np.ndarray | None = None,
    components: np.ndarray | None = None,
) -> PrototypeCorrectionField:
    """Convenience wrapper used by the CLI and small downstream experiments."""
    return PrototypeCorrectionField.fit(
        document_embeddings,
        query_embeddings,
        target_coordinates,
        n_prototypes=n_prototypes,
        top_m=top_m,
        temperature=temperature,
        ridge=ridge,
        random_state=random_state,
        mean=mean,
        components=components,
    )


def _subset_dataset(dataset, name: str, query_ids: set[str]) -> RetrievalDataset:
    queries = [sample for sample in dataset.queries if sample.query_id in query_ids]
    qrels = {sample.query_id: dataset.qrels[sample.query_id] for sample in queries}
    return RetrievalDataset(name, queries, dataset.corpus, qrels)


def _stable_split(dataset, train_fraction: float = 0.7):
    ordered = sorted(
        dataset.queries,
        key=lambda sample: hashlib.sha256(
            f"prototype-field-v1:{dataset.name}:{sample.query_id}".encode()
        ).digest(),
    )
    split = int(round(len(ordered) * train_fraction))
    train_ids = {sample.query_id for sample in ordered[:split]}
    test_ids = {sample.query_id for sample in ordered[split:]}
    return (
        _subset_dataset(dataset, f"{dataset.name}_train", train_ids),
        _subset_dataset(dataset, f"{dataset.name}_holdout", test_ids),
    )


def prepare_prototype_data(dataset_name: str, cache_root: Path, model_spec: dict):
    """Load cached documents and produce disjoint train/holdout query splits."""
    cache_dir = low_rank.find_cache_dir(cache_root, dataset_name, model_spec["cache_dir"])
    documents = np.load(cache_dir / "documents.npy", mmap_mode="r")
    if dataset_name == "fiqa":
        train_dataset = load_beir_dataset("fiqa", ROOT / "datasets", "train", False)
        test_dataset = load_beir_dataset("fiqa", ROOT / "datasets", "test", False)
        train_queries = np.load(cache_dir / "queries_train.npy", mmap_mode="r")
        test_queries = np.load(cache_dir / "queries.npy", mmap_mode="r")
        split_method = "official FiQA train/test qrels"
    else:
        full_dataset = load_beir_dataset(
            dataset_name,
            ROOT / "datasets",
            "test",
            False,
            "keep" if dataset_name == "arguana" else "error",
        )
        train_dataset, test_dataset = _stable_split(full_dataset)
        full_queries = np.load(cache_dir / "queries.npy", mmap_mode="r")
        row_index = {sample.query_id: row for row, sample in enumerate(full_dataset.queries)}
        train_rows = np.asarray([row_index[s.query_id] for s in train_dataset.queries])
        test_rows = np.asarray([row_index[s.query_id] for s in test_dataset.queries])
        train_queries = full_queries[train_rows]
        test_queries = full_queries[test_rows]
        split_method = f"stable 70/30 split of {dataset_name} test queries"
    if len(train_queries) != len(train_dataset.queries):
        raise ValueError(f"{dataset_name}: train query/cache count mismatch")
    if len(test_queries) != len(test_dataset.queries):
        raise ValueError(f"{dataset_name}: test query/cache count mismatch")
    return (
        train_dataset,
        test_dataset,
        np.asarray(train_queries),
        np.asarray(test_queries),
        documents,
        cache_dir,
        split_method,
    )


def run(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = args.output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    metadata = []

    for dataset_name in args.datasets:
        for model_key in args.models:
            model_spec = low_rank.MODEL_SPECS[model_key]
            model_label = model_spec["label"]
            (
                train_dataset,
                test_dataset,
                train_queries,
                test_queries,
                documents,
                cache_dir,
                split_method,
            ) = prepare_prototype_data(dataset_name, args.cache_root, model_spec)
            train_rows, train_deltas, train_missing = low_rank.build_oracle_deltas(
                train_dataset, train_queries, documents
            )
            # B is learned from train deltas only. It is not needed to fit the
            # field itself, but allows the saved V to be mapped back to deltas.
            mean, components = low_rank.fit_pca_subspace(
                train_deltas, max(args.rank, 1)
            )
            components = components[: args.rank]
            train_coordinates = project_correction_targets(
                train_deltas, mean, components
            )
            field = fit_field(
                documents,
                train_queries[train_rows],
                train_coordinates,
                n_prototypes=args.n_prototypes,
                top_m=args.top_m,
                temperature=args.temperature,
                ridge=args.ridge,
                random_state=args.random_state,
                mean=mean,
                components=components,
            )
            np.savez_compressed(
                model_dir / f"{dataset_name}_{model_key}.npz",
                prototypes=field.prototypes.astype(np.float32),
                values=field.values.astype(np.float32),
                mean=mean.astype(np.float32),
                components=components.astype(np.float32),
                temperature=np.asarray([args.temperature], dtype=np.float32),
                top_m=np.asarray([args.top_m], dtype=np.int64),
            )

            predicted_coordinates = field.predict_coordinates(test_queries)
            predicted_deltas = mean + predicted_coordinates @ components
            baseline = low_rank.evaluate(
                test_dataset, test_queries, documents, args.batch_size
            )
            for lambda_ in args.lambdas:
                corrected = field.apply(test_queries, lambda_=lambda_)
                metrics = low_rank.evaluate(
                    test_dataset, corrected, documents, args.batch_size
                )
                rows.append(
                    {
                        "dataset": dataset_name,
                        "model": model_label,
                        "lambda": float(lambda_),
                        **metrics,
                        "baseline_HitRate@10": baseline["HitRate@10"],
                    }
                )
            metadata.append(
                {
                    "dataset": dataset_name,
                    "model": model_label,
                    "split_method": split_method,
                    "cache_dir": str(cache_dir.relative_to(ROOT)),
                    "train_queries_total": len(train_dataset.queries),
                    "train_queries_used_for_field": len(train_rows),
                    "train_queries_missing_positive": train_missing,
                    "test_queries_total": len(test_dataset.queries),
                    "n_prototypes": args.n_prototypes,
                    "top_m": args.top_m,
                    "temperature": args.temperature,
                    "ridge": args.ridge,
                    "rank": args.rank,
                    "test_qrels_used_for_field": False,
                }
            )
            print(f"{dataset_name}/{model_label}: complete", flush=True)

    pd.DataFrame(rows).to_csv(
        args.output_dir / "prototype_metrics_by_lambda.csv", index=False
    )
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=ROOT / "cache")
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "results" / "prototype_correction"
    )
    parser.add_argument("--datasets", nargs="+", default=["fiqa", "arguana"])
    parser.add_argument(
        "--models", nargs="+", choices=low_rank.MODEL_SPECS, default=list(low_rank.MODEL_SPECS)
    )
    parser.add_argument("--n-prototypes", type=int, default=64)
    parser.add_argument("--top-m", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument(
        "--lambdas", nargs="+", type=float,
        default=[0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0],
    )
    parser.add_argument("--batch-size", type=int, default=128)
    return parser


if __name__ == "__main__":
    load_dotenv(ROOT / ".env")
    run(build_parser().parse_args())
