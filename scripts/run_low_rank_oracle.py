"""Evaluate leakage-free low-rank oracle corrections on held-out queries."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dense_retrieval.datasets import RetrievalDataset, load_beir_dataset
from dense_retrieval.embeddings import create_embedding_model
from dense_retrieval.evaluation.oracle import closest_positive_index
from dense_retrieval.evaluation.quality import compute_quality_metrics
from dense_retrieval.retrieval.exact import exact_search


MODEL_SPECS = {
    "qwen3": {
        "label": "Qwen3",
        "cache_dir": "Qwen__Qwen3-Embedding-0.6B",
        "config": "fiqa_qwen3_siliconflow.yaml",
    },
    "bge-m3": {
        "label": "BGE-M3",
        "cache_dir": "BAAI__bge-m3",
        "config": "fiqa_bge_m3_siliconflow.yaml",
    },
    "e5": {
        "label": "E5",
        "cache_dir": "intfloat__e5-base-v2",
        "config": "fiqa_e5_base_v2_openrouter.yaml",
    },
}
METRICS = [
    "HitRate@5",
    "HitRate@10",
    "Recall@5",
    "Recall@10",
    "MRR@10",
    "NDCG@10",
]
COLORS = {
    "baseline": "#666666",
    "full_oracle": "#111111",
    "rank_64": "#2f6f9f",
    "rank_128": "#c4513b",
    "rank_256": "#41844b",
}


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


def cached_encode_queries(model, texts, path: Path):
    if path.exists():
        array = np.load(path)
        if array.shape == (len(texts), model.dimension):
            return array.astype(np.float32)
    batch_size = int(getattr(model, "cache_batch_size", model.batch_size))
    parts_dir = path.parent / f"{path.stem}_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    parts = []
    for start in range(0, len(texts), batch_size):
        end = min(start + batch_size, len(texts))
        part_path = parts_dir / f"{start:09d}_{end:09d}.npy"
        part = np.load(part_path) if part_path.exists() else None
        if part is None or part.shape != (end - start, model.dimension):
            part = model.encode_queries(texts[start:end]).astype(np.float32)
            np.save(part_path, part)
        parts.append(part)
        print(f"train query encoding: {end}/{len(texts)}", flush=True)
    array = np.concatenate(parts)
    np.save(path, array)
    return array


def stable_arguana_split(dataset, train_fraction=0.7):
    ordered = sorted(
        dataset.queries,
        key=lambda sample: hashlib.sha256(
            f"arguana-low-rank-v1:{sample.query_id}".encode()
        ).digest(),
    )
    split_index = int(round(len(ordered) * train_fraction))
    train_ids = {sample.query_id for sample in ordered[:split_index]}
    test_ids = {sample.query_id for sample in ordered[split_index:]}

    def subset(name, ids):
        queries = [sample for sample in dataset.queries if sample.query_id in ids]
        qrels = {sample.query_id: dataset.qrels[sample.query_id] for sample in queries}
        return RetrievalDataset(name, queries, dataset.corpus, qrels)

    return subset("arguana_train", train_ids), subset("arguana_holdout", test_ids)


def rows_for_queries(full_dataset, subset_dataset):
    index = {
        sample.query_id: row for row, sample in enumerate(full_dataset.queries)
    }
    return np.asarray([index[sample.query_id] for sample in subset_dataset.queries])


def build_oracle_deltas(dataset, query_embeddings, document_embeddings):
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


def fit_pca_subspace(train_deltas, max_rank):
    mean = train_deltas.mean(axis=0)
    centered = train_deltas - mean
    _, _, components = np.linalg.svd(centered, full_matrices=False)
    return mean, components[:max_rank]


def project_deltas(deltas, mean, components, rank):
    basis = components[:rank]
    centered = deltas - mean
    return mean + (centered @ basis.T) @ basis


def apply_correction(query_embeddings, query_rows, deltas, lambda_):
    corrected = np.asarray(query_embeddings, dtype=np.float64).copy()
    corrected[query_rows] += float(lambda_) * deltas
    corrected /= np.maximum(np.linalg.norm(corrected, axis=1, keepdims=True), 1e-12)
    return corrected.astype(np.float32)


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


def evaluate(dataset, queries, documents, batch_size):
    rankings = top_k_document_ids(
        queries, documents, list(dataset.corpus), 10, batch_size
    )
    metrics, _ = compute_quality_metrics(
        [sample.query_id for sample in dataset.queries],
        dataset.qrels,
        rankings,
        cutoffs=(5, 10),
        available_document_ids=set(dataset.corpus),
    )
    return {metric: metrics[metric] for metric in METRICS}


def gain_retention(metric_value, baseline, oracle):
    denominator = oracle - baseline
    if abs(denominator) <= 1e-12:
        return np.nan
    return (metric_value - baseline) / denominator


def plot_curves(summary, dataset, model, output_path):
    subset = summary[
        (summary["dataset"] == dataset) & (summary["model"] == model)
    ]
    figure, axes = plt.subplots(2, 3, figsize=(12, 7.2), sharex=True)
    for axis, metric in zip(axes.flat, METRICS):
        for correction in COLORS:
            rows = subset[subset["correction"] == correction]
            axis.plot(
                rows["lambda"],
                rows[metric],
                marker="o",
                linewidth=2,
                linestyle="--" if correction == "baseline" else "-",
                color=COLORS[correction],
                label=correction.replace("_", " "),
            )
        axis.set_title(metric)
        axis.set_xlabel(r"Correction $\lambda$")
        axis.set_ylim(0, 1.01)
        axis.grid(True, alpha=0.25)
    axes[0, 0].legend(frameon=False, fontsize=8)
    figure.suptitle(f"{dataset} / {model}: held-out low-rank oracle")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_retention_curves(retention, dataset, model, output_path):
    subset = retention[
        (retention["dataset"] == dataset) & (retention["model"] == model)
    ]
    rank_colors = {64: "#2f6f9f", 128: "#c4513b", 256: "#41844b"}
    figure, axes = plt.subplots(2, 3, figsize=(12, 7.2), sharex=True, sharey=True)
    for axis, metric in zip(axes.flat, METRICS):
        metric_rows = subset[subset["metric"] == metric]
        for rank, color in rank_colors.items():
            rows = metric_rows[metric_rows["rank"] == rank]
            axis.plot(
                rows["lambda"],
                rows["ideal_gain_retention"],
                marker="o",
                linewidth=2,
                color=color,
                label=f"rank {rank}",
            )
        axis.axhline(1.0, color="#777777", linestyle="--", linewidth=1)
        axis.set_title(metric)
        axis.set_xlabel(r"Correction $\lambda$")
        axis.set_ylim(0, 1.08)
        axis.grid(True, alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        frameon=False,
        fontsize=8,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
    )
    figure.suptitle(f"{dataset} / {model}: ideal gain retention", y=0.995)
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def prepare_data(dataset_name, cache_root, model_spec):
    cache_dir = find_cache_dir(cache_root, dataset_name, model_spec["cache_dir"])
    documents = np.load(cache_dir / "documents.npy", mmap_mode="r")
    if dataset_name == "fiqa":
        train_dataset = load_beir_dataset("fiqa", ROOT / "datasets", "train", False)
        test_dataset = load_beir_dataset("fiqa", ROOT / "datasets", "test", False)
        config = yaml.safe_load(
            (ROOT / "configs" / model_spec["config"]).read_text(encoding="utf-8")
        )
        train_path = cache_dir / "queries_train.npy"
        if train_path.exists():
            train_queries = np.load(train_path, mmap_mode="r")
        else:
            model = create_embedding_model(config["model"])
            train_queries = cached_encode_queries(
                model, train_dataset.query_texts, train_path
            )
        test_queries = np.load(cache_dir / "queries.npy", mmap_mode="r")
        split_method = "official FiQA train/test qrels"
    else:
        full_dataset = load_beir_dataset(
            "arguana", ROOT / "datasets", "test", False, "keep"
        )
        train_dataset, test_dataset = stable_arguana_split(full_dataset)
        full_queries = np.load(cache_dir / "queries.npy", mmap_mode="r")
        train_queries = full_queries[rows_for_queries(full_dataset, train_dataset)]
        test_queries = full_queries[rows_for_queries(full_dataset, test_dataset)]
        split_method = "stable SHA-256 70/30 split of official ArguAna test queries"
    return (
        train_dataset,
        test_dataset,
        train_queries,
        test_queries,
        documents,
        cache_dir,
        split_method,
    )


def run(args):
    result_rows = []
    metadata = []
    subspace_dir = args.output_dir / "subspaces"
    subspace_dir.mkdir(parents=True, exist_ok=True)

    for dataset_name in args.datasets:
        for model_key in args.models:
            model_spec = MODEL_SPECS[model_key]
            model_label = model_spec["label"]
            (
                train_dataset,
                test_dataset,
                train_queries,
                test_queries,
                documents,
                cache_dir,
                split_method,
            ) = prepare_data(dataset_name, args.cache_root, model_spec)

            train_rows, train_deltas, train_missing = build_oracle_deltas(
                train_dataset, train_queries, documents
            )
            test_rows, test_deltas, test_missing = build_oracle_deltas(
                test_dataset, test_queries, documents
            )
            mean, components = fit_pca_subspace(train_deltas, max(args.ranks))
            np.savez_compressed(
                subspace_dir / f"{dataset_name}_{model_key}.npz",
                mean=mean.astype(np.float32),
                components=components.astype(np.float32),
            )
            projected = {
                rank: project_deltas(test_deltas, mean, components, rank)
                for rank in args.ranks
            }

            baseline_metrics = evaluate(
                test_dataset, test_queries, documents, args.batch_size
            )
            for lambda_ in args.lambdas:
                for correction, deltas in [
                    ("baseline", np.zeros_like(test_deltas)),
                    ("full_oracle", test_deltas),
                    *[(f"rank_{rank}", projected[rank]) for rank in args.ranks],
                ]:
                    corrected = (
                        np.asarray(test_queries, dtype=np.float32)
                        if correction == "baseline"
                        else apply_correction(
                            test_queries, test_rows, deltas, lambda_
                        )
                    )
                    metrics = (
                        baseline_metrics
                        if correction == "baseline"
                        else evaluate(
                            test_dataset, corrected, documents, args.batch_size
                        )
                    )
                    result_rows.append(
                        {
                            "dataset": dataset_name,
                            "model": model_label,
                            "lambda": lambda_,
                            "correction": correction,
                            **metrics,
                            "num_test_queries": len(test_dataset.queries),
                        }
                    )
                print(
                    f"{dataset_name}/{model_label}/lambda={lambda_:g}: complete",
                    flush=True,
                )

            train_ids = {sample.query_id for sample in train_dataset.queries}
            test_ids = {sample.query_id for sample in test_dataset.queries}
            overlap = train_ids.intersection(test_ids)
            if overlap:
                raise AssertionError(f"Train/test query leakage: {sorted(overlap)[:5]}")
            metadata.append(
                {
                    "dataset": dataset_name,
                    "model": model_label,
                    "split_method": split_method,
                    "cache_dir": str(cache_dir.relative_to(ROOT)),
                    "train_queries_total": len(train_dataset.queries),
                    "train_queries_used_for_subspace": len(train_rows),
                    "train_queries_missing_positive": train_missing,
                    "test_queries_total": len(test_dataset.queries),
                    "test_queries_with_oracle_delta": len(test_rows),
                    "test_queries_missing_positive": test_missing,
                    "train_test_query_id_overlap": len(overlap),
                    "subspace_fit_inputs": "train query-level delta_star only",
                    "test_delta_usage": "projection coefficients only",
                    "ranks": args.ranks,
                    "lambdas": args.lambdas,
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(result_rows)
    summary.to_csv(args.output_dir / "low_rank_metrics_by_lambda.csv", index=False)

    retention_rows = []
    for (dataset, model, lambda_), group in summary.groupby(
        ["dataset", "model", "lambda"]
    ):
        values = group.set_index("correction")
        for rank in args.ranks:
            for metric in METRICS:
                retention_rows.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "lambda": lambda_,
                        "rank": rank,
                        "metric": metric,
                        "baseline": values.loc["baseline", metric],
                        "full_oracle": values.loc["full_oracle", metric],
                        "low_rank_oracle": values.loc[f"rank_{rank}", metric],
                        "ideal_gain_retention": gain_retention(
                            values.loc[f"rank_{rank}", metric],
                            values.loc["baseline", metric],
                            values.loc["full_oracle", metric],
                        ),
                    }
                )
    retention = pd.DataFrame(retention_rows)
    retention.to_csv(args.output_dir / "ideal_gain_retention_by_lambda.csv", index=False)

    hit10 = summary.pivot(
        index=["dataset", "model", "lambda"],
        columns="correction",
        values="HitRate@10",
    ).reset_index()
    hit10 = hit10[
        [
            "dataset",
            "model",
            "lambda",
            "baseline",
            "full_oracle",
            *[f"rank_{rank}" for rank in args.ranks],
        ]
    ]
    hit10.to_csv(args.output_dir / "hit10_curve.csv", index=False)
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    for dataset in args.datasets:
        for model_key in args.models:
            model = MODEL_SPECS[model_key]["label"]
            plot_curves(
                summary,
                dataset,
                model,
                args.output_dir / f"{dataset}_{model_key}_curves.png",
            )
            plot_retention_curves(
                retention,
                dataset,
                model,
                args.output_dir / f"{dataset}_{model_key}_retention.png",
            )

    print("\nHitRate@10 curves:")
    print(hit10.round(4).to_string(index=False))


if __name__ == "__main__":
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, default=ROOT / "cache")
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "results" / "low_rank_oracle"
    )
    parser.add_argument("--datasets", nargs="+", default=["fiqa", "arguana"])
    parser.add_argument(
        "--models", nargs="+", choices=MODEL_SPECS, default=list(MODEL_SPECS)
    )
    parser.add_argument("--ranks", nargs="+", type=int, default=[64, 128, 256])
    parser.add_argument(
        "--lambdas",
        nargs="+",
        type=float,
        default=[0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0],
    )
    parser.add_argument("--batch-size", type=int, default=128)
    run(parser.parse_args())
