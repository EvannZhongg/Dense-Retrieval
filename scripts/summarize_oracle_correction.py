"""Audit and summarize oracle query correction metrics by lambda."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dense_retrieval.datasets import load_beir_dataset
from dense_retrieval.evaluation.oracle import closest_positive_index
from dense_retrieval.evaluation.quality import compute_quality_metrics
from dense_retrieval.retrieval.exact import exact_search


MODEL_SPECS = {
    "qwen3": (
        "Qwen3",
        "Qwen__Qwen3-Embedding-0.6B",
        "qwen3-embedding-0.6b",
    ),
    "bge-m3": ("BGE-M3", "BAAI__bge-m3", "bge-m3"),
    "e5": ("E5", "intfloat__e5-base-v2", "e5-base-v2"),
}
COLORS = {"Qwen3": "#2f6f9f", "BGE-M3": "#c4513b", "E5": "#41844b"}
METRICS = [
    "HitRate@5",
    "HitRate@10",
    "Recall@5",
    "Recall@10",
    "MRR@10",
    "NDCG@10",
]


def find_cache_dir(cache_root: Path, dataset: str, model_dir: str) -> Path:
    root = cache_root / dataset / model_dir
    candidates = (
        sorted(
            path
            for path in root.iterdir()
            if (path / "queries.npy").exists()
            and (path / "documents.npy").exists()
        )
        if root.exists()
        else []
    )
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one complete cache under {root}, found {len(candidates)}"
        )
    return candidates[0]


def select_oracle_positives(dataset, query_embeddings, document_embeddings):
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


def apply_oracle_correction(query_embeddings, positives, available, lambda_):
    corrected = np.asarray(query_embeddings, dtype=np.float32).copy()
    corrected[available] = (
        corrected[available] * (1.0 - lambda_)
        + positives[available] * lambda_
    )
    norms = np.linalg.norm(corrected, axis=1, keepdims=True)
    return corrected / np.maximum(norms, 1e-12)


def top_k_document_ids(
    query_embeddings, document_embeddings, document_ids, k=10, batch_size=128
):
    rankings = []
    document_ids = np.asarray(document_ids, dtype=object)
    for start in range(0, len(query_embeddings), batch_size):
        indices, _ = exact_search(
            query_embeddings[start : start + batch_size], document_embeddings, k
        )
        rankings.extend(document_ids[indices].tolist())
    return rankings


def validate_against_parquet(existing, per_query, lambda_, dataset, model):
    expected = existing[np.isclose(existing["lambda"], lambda_)].copy()
    expected["query_id"] = expected["query_id"].astype(str)
    actual = per_query.copy()
    actual["query_id"] = actual["query_id"].astype(str)
    joined = expected.set_index("query_id").join(
        actual.set_index("query_id"), how="inner", rsuffix="_recomputed"
    )
    if len(joined) != len(per_query):
        raise AssertionError(f"{dataset}/{model}/lambda={lambda_}: query mismatch")
    comparisons = {
        "entered_top5": "hit_at_5",
        "entered_top10": "hit_at_10",
        "recall_at_5": "recall_at_5_recomputed",
        "recall_at_10": "recall_at_10_recomputed",
    }
    for old_column, new_column in comparisons.items():
        old = joined[old_column].astype(float).to_numpy()
        new = joined[new_column].astype(float).to_numpy()
        if not np.allclose(old, new, atol=1e-7):
            mismatch = int(np.count_nonzero(~np.isclose(old, new, atol=1e-7)))
            raise AssertionError(
                f"{dataset}/{model}/lambda={lambda_}: {old_column} has "
                f"{mismatch} mismatches"
            )


def plot_metrics(summary, dataset, output_path):
    figure, axes = plt.subplots(2, 3, figsize=(12, 7.2), sharex=True)
    subset = summary[summary["dataset"] == dataset]
    for axis, metric in zip(axes.flat, METRICS):
        for model in COLORS:
            rows = subset[subset["model"] == model]
            axis.plot(
                rows["lambda"],
                rows[metric],
                marker="o",
                linewidth=2,
                color=COLORS[model],
                label=model,
            )
        axis.set_title(metric)
        axis.set_xlabel(r"Oracle correction $\lambda$")
        axis.set_ylim(0, 1.01)
        axis.grid(True, alpha=0.25)
    axes[0, 0].legend(frameon=False)
    figure.suptitle(f"{dataset}: ideal query-side correction")
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def run(args):
    rows = []
    for dataset_name in args.datasets:
        dataset = load_beir_dataset(
            dataset_name,
            args.dataset_root,
            "test",
            False,
            missing_relevant_policy="keep",
        )
        query_ids = [sample.query_id for sample in dataset.queries]
        document_ids = list(dataset.corpus)

        for model_key in args.models:
            model_label, model_cache_dir, model_result_dir = MODEL_SPECS[model_key]
            cache_dir = find_cache_dir(
                args.cache_root, dataset_name, model_cache_dir
            )
            query_embeddings = np.load(cache_dir / "queries.npy", mmap_mode="r")
            document_embeddings = np.load(
                cache_dir / "documents.npy", mmap_mode="r"
            )
            parquet_path = (
                args.results_root
                / dataset_name
                / model_result_dir
                / "oracle_correction.parquet"
            )
            existing = pd.read_parquet(parquet_path)
            existing_lambdas = sorted(
                float(value) for value in existing["lambda"].unique()
            )
            lambdas = sorted(set(existing_lambdas + args.extra_lambdas))
            positives, available = select_oracle_positives(
                dataset, query_embeddings, document_embeddings
            )

            for lambda_ in lambdas:
                corrected = apply_oracle_correction(
                    query_embeddings, positives, available, lambda_
                )
                rankings = top_k_document_ids(
                    corrected,
                    document_embeddings,
                    document_ids,
                    k=10,
                    batch_size=args.batch_size,
                )
                metrics, per_query = compute_quality_metrics(
                    query_ids,
                    dataset.qrels,
                    rankings,
                    cutoffs=(5, 10),
                    available_document_ids=set(document_ids),
                )
                from_existing_parquet = any(
                    np.isclose(lambda_, value) for value in existing_lambdas
                )
                if from_existing_parquet:
                    validate_against_parquet(
                        existing, per_query, lambda_, dataset_name, model_label
                    )
                rows.append(
                    {
                        "dataset": dataset_name,
                        "model": model_label,
                        "lambda": lambda_,
                        "lambda_in_existing_parquet": from_existing_parquet,
                        **{metric: metrics[metric] for metric in METRICS},
                        "num_queries": metrics["num_queries"],
                        "correction_available_queries": int(available.sum()),
                    }
                )
                print(
                    f"{dataset_name}/{model_label}/lambda={lambda_:g}: "
                    f"{'verified' if from_existing_parquet else 'ceiling probe'}",
                    flush=True,
                )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(rows)
    summary.to_csv(args.output_dir / "oracle_metrics_by_lambda.csv", index=False)

    baseline = summary[summary["lambda"] == 0].set_index(["dataset", "model"])
    gains = summary.copy()
    for metric in METRICS:
        gains[f"delta_{metric}"] = [
            row[metric] - baseline.loc[(row["dataset"], row["model"]), metric]
            for _, row in gains.iterrows()
        ]
    gains.to_csv(args.output_dir / "oracle_gain_over_baseline.csv", index=False)

    best_rows = []
    for (dataset, model), group in summary.groupby(["dataset", "model"]):
        for metric in METRICS:
            best_index = group[metric].idxmax()
            best = summary.loc[best_index]
            base = baseline.loc[(dataset, model), metric]
            best_rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "metric": metric,
                    "baseline": float(base),
                    "best_value": float(best[metric]),
                    "absolute_gain": float(best[metric] - base),
                    "best_lambda": float(best["lambda"]),
                    "best_lambda_in_existing_parquet": bool(
                        best["lambda_in_existing_parquet"]
                    ),
                }
            )
    best = pd.DataFrame(best_rows)
    best.to_csv(args.output_dir / "oracle_best_by_metric.csv", index=False)

    for dataset in args.datasets:
        plot_metrics(
            summary,
            dataset,
            args.output_dir / f"{dataset}_oracle_metrics_vs_lambda.png",
        )

    print("\nMetrics at lambda=1.0 (full move to the selected positive):")
    print(
        summary[np.isclose(summary["lambda"], 1.0)]
        .set_index(["dataset", "model"])[METRICS]
        .round(4)
        .to_string()
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "datasets")
    parser.add_argument("--cache-root", type=Path, default=ROOT / "cache")
    parser.add_argument("--results-root", type=Path, default=ROOT / "results")
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "results" / "oracle_summary"
    )
    parser.add_argument("--datasets", nargs="+", default=["fiqa", "arguana"])
    parser.add_argument(
        "--models", nargs="+", choices=MODEL_SPECS, default=list(MODEL_SPECS)
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--extra-lambdas",
        nargs="+",
        type=float,
        default=[0.3, 0.5, 0.75, 1.0],
        help="Additional ceiling probes beyond lambdas stored in the parquet files",
    )
    run(parser.parse_args())
