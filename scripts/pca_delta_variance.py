"""PCA of query-level oracle correction targets from cached embeddings."""
from __future__ import annotations

import argparse
import json
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


MODEL_SPECS = {
    "qwen3": ("Qwen3", "Qwen__Qwen3-Embedding-0.6B"),
    "bge-m3": ("BGE-M3", "BAAI__bge-m3"),
    "e5": ("E5", "intfloat__e5-base-v2"),
}
COLORS = {"Qwen3": "#2f6f9f", "BGE-M3": "#c4513b", "E5": "#41844b"}
VARIANT_TITLES = {
    "delta_star": r"$\Delta^* = d^* - q$",
    "delta_star_q_orthogonal_unit": r"Unit $q$-orthogonal direction",
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


def build_query_level_deltas(dataset, query_embeddings, document_embeddings):
    """Build one closest-positive target and one unit tangent direction per query."""
    document_index = {
        doc_id: index for index, doc_id in enumerate(dataset.corpus)
    }
    deltas = []
    orthogonal_directions = []
    missing_pairs = []

    for query_index, sample in enumerate(dataset.queries):
        query_id = sample.query_id
        positive_index = closest_positive_index(
            query_id,
            dataset.qrels,
            query_embeddings[query_index],
            document_embeddings,
            document_index,
        )
        if positive_index is None:
            missing_pairs.extend(
                (query_id, str(document_id))
                for document_id, relevance in dataset.qrels[query_id].items()
                if int(relevance) > 0 and str(document_id) not in document_index
            )
            continue

        query = np.asarray(query_embeddings[query_index], dtype=np.float64)
        positive = np.asarray(document_embeddings[positive_index], dtype=np.float64)
        delta = positive - query
        query_squared_norm = float(query @ query)
        orthogonal = delta - (float(query @ delta) / query_squared_norm) * query
        orthogonal_norm = np.linalg.norm(orthogonal)
        if orthogonal_norm <= 1e-12:
            raise ValueError(f"Query {query_id} has a degenerate tangent direction")

        deltas.append(delta)
        orthogonal_directions.append(orthogonal / orthogonal_norm)

    return {
        "delta_star": np.asarray(deltas, dtype=np.float64),
        "delta_star_q_orthogonal_unit": np.asarray(
            orthogonal_directions, dtype=np.float64
        ),
    }, missing_pairs


def pca_variance_spectrum(samples: np.ndarray):
    if samples.ndim != 2 or samples.shape[0] < 2:
        raise ValueError("PCA requires a 2-D array with at least two samples")
    centered = samples - samples.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
    variances = singular_values**2 / (samples.shape[0] - 1)
    explained_ratio = variances / variances.sum()
    return singular_values, explained_ratio, np.cumsum(explained_ratio)


def plot_spectra(spectrum: pd.DataFrame, output_path: Path, ranks: list[int]):
    datasets = list(dict.fromkeys(spectrum["dataset"]))
    variants = list(dict.fromkeys(spectrum["variant"]))
    figure, axes = plt.subplots(
        len(variants), len(datasets), figsize=(12, 8.3), sharex=True, sharey=True
    )
    axes = np.asarray(axes).reshape(len(variants), len(datasets))

    for row_index, variant in enumerate(variants):
        for column_index, dataset in enumerate(datasets):
            axis = axes[row_index, column_index]
            subset = spectrum[
                (spectrum["dataset"] == dataset)
                & (spectrum["variant"] == variant)
            ]
            for model_key in MODEL_SPECS:
                label = MODEL_SPECS[model_key][0]
                rows = subset[subset["model"] == label]
                axis.plot(
                    rows["rank"],
                    rows["cumulative_explained_variance_ratio"],
                    color=COLORS[label],
                    linewidth=2,
                    label=label,
                )
                marked = rows[rows["rank"].isin(ranks)]
                axis.scatter(
                    marked["rank"],
                    marked["cumulative_explained_variance_ratio"],
                    color=COLORS[label],
                    s=19,
                    zorder=3,
                )
            axis.set_xscale("log", base=2)
            axis.set_xticks(ranks, labels=[str(rank) for rank in ranks])
            axis.set_xlim(min(ranks) * 0.8, max(ranks) * 1.2)
            axis.set_ylim(0, 1.01)
            axis.grid(True, alpha=0.25)
            axis.set_title(f"{dataset}: {VARIANT_TITLES[variant]}")
            if row_index == len(variants) - 1:
                axis.set_xlabel("PCA rank r")
            if column_index == 0:
                axis.set_ylabel("Cumulative explained variance ratio")
    axes[0, -1].legend(frameon=False, loc="lower right")
    figure.suptitle("Query-level oracle correction targets")
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def run(args):
    spectrum_rows = []
    summary_rows = []
    metadata = []

    for dataset_name in args.datasets:
        dataset = load_beir_dataset(
            dataset_name,
            args.dataset_root,
            "test",
            False,
            missing_relevant_policy="keep",
        )
        for model_key in args.models:
            model_label, model_dir = MODEL_SPECS[model_key]
            cache_dir = find_cache_dir(args.cache_root, dataset_name, model_dir)
            query_embeddings = np.load(cache_dir / "queries.npy", mmap_mode="r")
            document_embeddings = np.load(
                cache_dir / "documents.npy", mmap_mode="r"
            )
            if len(query_embeddings) != len(dataset.queries):
                raise ValueError(f"{dataset_name}/{model_label}: query count mismatch")
            if len(document_embeddings) != len(dataset.corpus):
                raise ValueError(f"{dataset_name}/{model_label}: document count mismatch")

            variants, missing_pairs = build_query_level_deltas(
                dataset, query_embeddings, document_embeddings
            )
            singular_values_by_variant = {}
            for variant, samples in variants.items():
                singular_values, ratio, cumulative = pca_variance_spectrum(samples)
                singular_values_by_variant[variant] = singular_values.tolist()
                for component, (value, cumulative_value) in enumerate(
                    zip(ratio, cumulative), start=1
                ):
                    spectrum_rows.append(
                        {
                            "dataset": dataset_name,
                            "model": model_label,
                            "variant": variant,
                            "rank": component,
                            "explained_variance_ratio": float(value),
                            "cumulative_explained_variance_ratio": float(
                                cumulative_value
                            ),
                        }
                    )
                for rank in args.ranks:
                    if rank > len(cumulative):
                        raise ValueError(
                            f"Rank {rank} exceeds {dataset_name}/{model_label}/"
                            f"{variant} PCA rank {len(cumulative)}"
                        )
                    summary_rows.append(
                        {
                            "dataset": dataset_name,
                            "model": model_label,
                            "variant": variant,
                            "rank": rank,
                            "cumulative_explained_variance_ratio": float(
                                cumulative[rank - 1]
                            ),
                        }
                    )

            metadata.append(
                {
                    "dataset": dataset_name,
                    "model": model_label,
                    "cache_dir": str(cache_dir.relative_to(ROOT)),
                    "embedding_dimension": int(query_embeddings.shape[1]),
                    "queries_used": int(variants["delta_star"].shape[0]),
                    "queries_missing_positive_document": len(missing_pairs),
                    "missing_pairs": [
                        {"query_id": query_id, "document_id": document_id}
                        for query_id, document_id in missing_pairs
                    ],
                    "orthogonal_variant": (
                        "unit_l2(delta_star - projection_of_delta_star_onto_q)"
                    ),
                    "singular_values": singular_values_by_variant,
                }
            )
            print(
                f"{dataset_name}/{model_label}: "
                f"{len(variants['delta_star'])} query-level targets, "
                f"dimension={query_embeddings.shape[1]}, missing={len(missing_pairs)}",
                flush=True,
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    spectrum = pd.DataFrame(spectrum_rows)
    summary = pd.DataFrame(summary_rows)
    spectrum.to_csv(args.output_dir / "pca_spectrum.csv", index=False)
    summary.to_csv(args.output_dir / "rank_summary.csv", index=False)
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    plot_spectra(
        spectrum, args.output_dir / "explained_variance_vs_rank.png", args.ranks
    )
    print("\nCumulative explained variance ratio:")
    print(
        summary.pivot(
            index=["variant", "dataset", "model"],
            columns="rank",
            values="cumulative_explained_variance_ratio",
        )
        .round(4)
        .to_string()
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "datasets")
    parser.add_argument("--cache-root", type=Path, default=ROOT / "cache")
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "results" / "delta_pca"
    )
    parser.add_argument("--datasets", nargs="+", default=["fiqa", "arguana"])
    parser.add_argument(
        "--models", nargs="+", choices=MODEL_SPECS, default=list(MODEL_SPECS)
    )
    parser.add_argument(
        "--ranks", nargs="+", type=int, default=[16, 32, 64, 128, 256]
    )
    run(parser.parse_args())
