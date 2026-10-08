"""Audit whether a global delta-PCA action space hides ranking utility.

Every fold holds out one complete corpus. Subspaces are fitted with labeled
training queries from the remaining corpora, while the held-out qrels are used
only to measure privileged oracle capacity. This is an upper-bound diagnostic,
not a deployable correction policy.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dense_retrieval.analysis.action_space import (  # noqa: E402
    fit_svd_subspace,
    optimize_ranking_oracle,
    project_to_subspace,
    projection_statistics,
    random_subspace,
    ranking_boundary_directions,
    tangent_projection,
)
from dense_retrieval.analysis.query_correction import (  # noqa: E402
    apply_correction,
    build_oracle_deltas,
    gain_retention,
)
from dense_retrieval.analysis.study_data import prepare_three_way  # noqa: E402
from dense_retrieval.analysis.ranking_correction import build_rank_records  # noqa: E402
from dense_retrieval.embeddings import MODEL_SPECS  # noqa: E402
from dense_retrieval.evaluation.ranking import evaluate_retrieval  # noqa: E402
from dense_retrieval.retrieval.exact import clear_cuda_document_cache  # noqa: E402


BASIS_METHODS = (
    "delta_svd",
    "tangent_delta_svd",
    "ranking_boundary_svd",
    "random_subspace",
)


def _markdown_table(frame: pd.DataFrame) -> str:
    """Format a small numeric summary without an optional tabulate dependency."""
    columns = list(frame.columns)
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for values in frame.itertuples(index=False, name=None):
        cells = [f"{value:.4f}" if isinstance(value, float) else str(value) for value in values]
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def _prepare_corpus(dataset_name: str, model_key: str, args: argparse.Namespace):
    (
        train_dataset,
        _dev_dataset,
        test_dataset,
        train_queries,
        _dev_queries,
        test_queries,
        documents,
        cache_dir,
        split,
    ) = prepare_three_way(
        dataset_name,
        model_key,
        args.cache_root,
        datasets_root=ROOT / "datasets",
        configs_root=ROOT / "configs",
    )
    train_rows, train_deltas, train_missing = build_oracle_deltas(
        train_dataset, train_queries, documents
    )
    test_rows, test_deltas, test_missing = build_oracle_deltas(
        test_dataset, test_queries, documents
    )
    _, boundary_directions = ranking_boundary_directions(
        train_dataset,
        train_queries,
        documents,
        hard_negatives=args.hard_negatives,
        temperature=args.temperature,
    )
    return {
        "train_deltas": train_deltas,
        "train_tangent_deltas": tangent_projection(
            train_deltas, train_queries[train_rows]
        ),
        "train_boundary_directions": boundary_directions,
        "test_dataset": test_dataset,
        "test_queries": test_queries,
        "test_rows": test_rows,
        "test_deltas": test_deltas,
        "documents": documents,
        "metadata": {
            "split": split,
            "cache_dir": str(cache_dir.relative_to(ROOT)),
            "train_queries": len(train_dataset.queries),
            "train_oracle_deltas": len(train_deltas),
            "train_boundary_directions": len(boundary_directions),
            "train_missing_positive": train_missing,
            "test_queries": len(test_dataset.queries),
            "test_oracle_deltas": len(test_deltas),
            "test_missing_positive": test_missing,
        },
    }


def _fit_fold_bases(corpora, training_names, max_rank, random_state):
    deltas = np.concatenate(
        [corpora[name]["train_deltas"] for name in training_names], axis=0
    )
    tangent_deltas = np.concatenate(
        [corpora[name]["train_tangent_deltas"] for name in training_names], axis=0
    )
    boundary = np.concatenate(
        [corpora[name]["train_boundary_directions"] for name in training_names],
        axis=0,
    )
    dimension = deltas.shape[1]
    return {
        "delta_svd": fit_svd_subspace(deltas, max_rank),
        "tangent_delta_svd": fit_svd_subspace(tangent_deltas, max_rank),
        "ranking_boundary_svd": fit_svd_subspace(boundary, max_rank),
        "random_subspace": random_subspace(dimension, max_rank, random_state),
    }, {
        "pooled_train_deltas": len(deltas),
        "pooled_train_boundary_directions": len(boundary),
    }


def _best_ndcg_rows(metrics: pd.DataFrame) -> pd.DataFrame:
    keys = ["heldout_corpus", "model", "method", "rank"]
    ordered = metrics.sort_values(
        keys + ["NDCG@10", "lambda"],
        ascending=[True, True, True, True, False, True],
    )
    best = ordered.groupby(keys, as_index=False).first()
    baselines = (
        best[best["method"] == "baseline"]
        .set_index(["heldout_corpus", "model"])["NDCG@10"]
        .rename("baseline_ndcg")
    )
    full = (
        best[best["method"] == "full_oracle"]
        .set_index(["heldout_corpus", "model"])["NDCG@10"]
        .rename("full_oracle_ndcg")
    )
    best = best.join(baselines, on=["heldout_corpus", "model"])
    best = best.join(full, on=["heldout_corpus", "model"])
    best["oracle_ndcg_gain_retention"] = [
        gain_retention(value, baseline, oracle)
        for value, baseline, oracle in zip(
            best["NDCG@10"], best["baseline_ndcg"], best["full_oracle_ndcg"]
        )
    ]
    return best


def _ranking_oracle_retention(frame: pd.DataFrame) -> pd.DataFrame:
    baselines = (
        frame[frame["method"] == "baseline"]
        .set_index(["heldout_corpus", "model"])["NDCG@10"]
        .rename("baseline_ndcg")
    )
    full = (
        frame[frame["method"] == "full_ambient"]
        .set_index(["heldout_corpus", "model"])["NDCG@10"]
        .rename("full_ambient_ndcg")
    )
    result = frame.join(baselines, on=["heldout_corpus", "model"])
    result = result.join(full, on=["heldout_corpus", "model"])
    result["ranking_oracle_ndcg_gain_retention"] = [
        gain_retention(value, baseline, oracle)
        for value, baseline, oracle in zip(
            result["NDCG@10"], result["baseline_ndcg"], result["full_ambient_ndcg"]
        )
    ]
    return result


def _write_findings(
    output_dir: Path,
    best: pd.DataFrame,
    ranking_oracles: pd.DataFrame,
    geometry: pd.DataFrame,
    rank: int,
) -> None:
    subset = best[(best["rank"] == rank) & best["method"].isin(BASIS_METHODS)]
    summary = (
        subset.groupby("method", as_index=False)
        .agg(
            mean_ndcg_gain_retention=("oracle_ndcg_gain_retention", "mean"),
            mean_best_ndcg=("NDCG@10", "mean"),
            mean_oracle_lambda=("lambda", "mean"),
        )
        .sort_values("mean_ndcg_gain_retention", ascending=False)
    )
    paired = subset.pivot_table(
        index=["heldout_corpus", "model"],
        columns="method",
        values="oracle_ndcg_gain_retention",
    )
    boundary_wins = int(
        (
            paired["ranking_boundary_svd"]
            > paired["delta_svd"]
        ).sum()
    )
    joined = geometry.merge(
        best[
            [
                "heldout_corpus",
                "model",
                "method",
                "rank",
                "oracle_ndcg_gain_retention",
            ]
        ],
        on=["heldout_corpus", "model", "method", "rank"],
        how="inner",
    )
    correlation = joined["energy_retention"].corr(
        joined["oracle_ndcg_gain_retention"], method="spearman"
    )
    oracle_subset = ranking_oracles[
        (ranking_oracles["rank"] == rank)
        & ranking_oracles["method"].isin(BASIS_METHODS)
    ]
    oracle_summary = (
        oracle_subset.groupby("method", as_index=False)
        .agg(
            mean_ranking_oracle_gain_retention=(
                "ranking_oracle_ndcg_gain_retention",
                "mean",
            ),
            mean_ranking_oracle_ndcg=("NDCG@10", "mean"),
        )
        .sort_values("mean_ranking_oracle_gain_retention", ascending=False)
    )
    oracle_paired = oracle_subset.pivot_table(
        index=["heldout_corpus", "model"],
        columns="method",
        values="ranking_oracle_ndcg_gain_retention",
    )
    oracle_boundary_wins = int(
        (oracle_paired["ranking_boundary_svd"] > oracle_paired["delta_svd"]).sum()
    )
    lines = [
        "# Action-space audit",
        "",
        "This is a privileged capacity diagnostic. Every basis is fitted without the",
        "held-out corpus, but projection coefficients and lambda selection use held-out",
        "qrels only to measure the best utility available inside each action space.",
        "",
        f"## Rank {rank} projected nearest-positive oracle",
        "",
        _markdown_table(summary),
        "",
        f"## Rank {rank} direct ranking oracle",
        "",
        _markdown_table(oracle_summary),
        "",
        "## Diagnostics",
        "",
        f"- Ranking-boundary SVD beats raw delta SVD in {boundary_wins}/{len(paired)} held-out folds at rank {rank}.",
        f"- With direct coordinate optimization, ranking-boundary SVD beats raw delta SVD in {oracle_boundary_wins}/{len(oracle_paired)} folds at rank {rank}.",
        f"- Across methods and ranks, Spearman correlation between delta energy retention and best NDCG gain retention is {correlation:.4f}.",
        "- A high reconstruction score is not sufficient evidence of ranking utility; the fixed-lambda curves remain the primary audit trail.",
        "",
    ]
    (output_dir / "findings.md").write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    if len(args.datasets) < 2:
        raise ValueError("at least two corpora are required for leave-one-corpus-out")
    if len(set(args.datasets)) != len(args.datasets):
        raise ValueError("datasets must be unique")
    if sorted(set(args.ranks)) != sorted(args.ranks) or min(args.ranks) < 1:
        raise ValueError("ranks must be unique positive integers")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metric_rows = []
    geometry_rows = []
    ranking_oracle_rows = []
    metadata = []

    for model_index, model_key in enumerate(args.models):
        print(f"Preparing {model_key}", flush=True)
        corpora = {
            name: _prepare_corpus(name, model_key, args) for name in args.datasets
        }
        model_label = MODEL_SPECS[model_key]["label"]
        for heldout_index, heldout_name in enumerate(args.datasets):
            training_names = [name for name in args.datasets if name != heldout_name]
            bases, counts = _fit_fold_bases(
                corpora,
                training_names,
                max(args.ranks),
                args.random_state + model_index * 10007 + heldout_index * 1009,
            )
            heldout = corpora[heldout_name]
            baseline_metrics = evaluate_retrieval(
                heldout["test_dataset"],
                heldout["test_queries"],
                heldout["documents"],
                args.batch_size,
            )
            projections = {}
            for method, components in bases.items():
                for rank in args.ranks:
                    projected = project_to_subspace(
                        heldout["test_deltas"], components, rank
                    )
                    projections[(method, rank)] = projected
                    geometry_rows.append(
                        {
                            "heldout_corpus": heldout_name,
                            "model": model_label,
                            "method": method,
                            "rank": rank,
                            **projection_statistics(
                                heldout["test_deltas"], projected
                            ),
                        }
                    )

            ranking_oracle_rows.append(
                {
                    "heldout_corpus": heldout_name,
                    "model": model_label,
                    "method": "baseline",
                    "rank": 0,
                    "rank_loss": np.nan,
                    "mean_delta_norm": 0.0,
                    **baseline_metrics,
                }
            )
            rank_records = build_rank_records(
                heldout["test_dataset"],
                heldout["test_queries"],
                heldout["documents"],
                args.hard_negatives,
            )
            oracle_spaces = [("full_ambient", 0, None)] + [
                (method, rank, bases[method][:rank])
                for method in BASIS_METHODS
                for rank in args.ranks
            ]
            for method, rank, components in oracle_spaces:
                oracle_rows, oracle_deltas, rank_loss = optimize_ranking_oracle(
                    heldout["test_queries"],
                    heldout["documents"],
                    rank_records,
                    components,
                    steps=args.oracle_steps,
                    learning_rate=args.oracle_learning_rate,
                    penalty=args.oracle_penalty,
                    temperature=args.temperature,
                    device=args.device,
                )
                corrected = apply_correction(
                    heldout["test_queries"], oracle_rows, oracle_deltas, 1.0
                )
                scores = evaluate_retrieval(
                    heldout["test_dataset"],
                    corrected,
                    heldout["documents"],
                    args.batch_size,
                )
                ranking_oracle_rows.append(
                    {
                        "heldout_corpus": heldout_name,
                        "model": model_label,
                        "method": method,
                        "rank": rank,
                        "rank_loss": rank_loss,
                        "mean_delta_norm": float(
                            np.mean(np.linalg.norm(oracle_deltas, axis=1))
                        ),
                        **scores,
                    }
                )
            print(f"{model_key}/{heldout_name}: ranking oracles complete", flush=True)
            for lambda_ in args.lambdas:
                variants = [
                    ("baseline", 0, np.zeros_like(heldout["test_deltas"])),
                    ("full_oracle", 0, heldout["test_deltas"]),
                    *[
                        (method, rank, projections[(method, rank)])
                        for method in BASIS_METHODS
                        for rank in args.ranks
                    ],
                ]
                for method, rank, deltas in variants:
                    if method == "baseline":
                        scores = baseline_metrics
                    else:
                        corrected = apply_correction(
                            heldout["test_queries"],
                            heldout["test_rows"],
                            deltas,
                            lambda_,
                        )
                        scores = evaluate_retrieval(
                            heldout["test_dataset"],
                            corrected,
                            heldout["documents"],
                            args.batch_size,
                        )
                    metric_rows.append(
                        {
                            "heldout_corpus": heldout_name,
                            "model": model_label,
                            "method": method,
                            "rank": rank,
                            "lambda": lambda_,
                            **scores,
                        }
                    )
                print(
                    f"{model_key}/{heldout_name}/lambda={lambda_:g}: complete",
                    flush=True,
                )
            metadata.append(
                {
                    "model": model_label,
                    "heldout_corpus": heldout_name,
                    "training_corpora": training_names,
                    **counts,
                    "heldout": heldout["metadata"],
                }
            )
        clear_cuda_document_cache()

    metrics = pd.DataFrame(metric_rows)
    geometry = pd.DataFrame(geometry_rows)
    ranking_oracles = _ranking_oracle_retention(pd.DataFrame(ranking_oracle_rows))
    best = _best_ndcg_rows(metrics)
    metrics.to_csv(args.output_dir / "action_space_metrics.csv", index=False)
    geometry.to_csv(args.output_dir / "action_space_geometry.csv", index=False)
    best.to_csv(args.output_dir / "best_ndcg_by_method.csv", index=False)
    ranking_oracles.to_csv(
        args.output_dir / "ranking_oracle_metrics.csv", index=False
    )
    (args.output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "protocol": "leave-one-corpus-out privileged action-space capacity audit",
                "heldout_qrels_used_for_basis_fit": False,
                "heldout_qrels_used_for_projection_and_oracle_lambda": True,
                "basis_methods": list(BASIS_METHODS),
                "ranks": args.ranks,
                "lambdas": args.lambdas,
                "hard_negatives": args.hard_negatives,
                "temperature": args.temperature,
                "ranking_oracle_steps": args.oracle_steps,
                "ranking_oracle_learning_rate": args.oracle_learning_rate,
                "ranking_oracle_penalty": args.oracle_penalty,
                "ranking_oracle_device": args.device,
                "folds": metadata,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_findings(
        args.output_dir,
        best,
        ranking_oracles,
        geometry,
        rank=32 if 32 in args.ranks else args.ranks[0],
    )
    print(best.round(4).to_string(index=False), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=ROOT / "cache")
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "results" / "action_space_audit"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["fiqa", "arguana", "scifact", "nfcorpus"],
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_SPECS,
        default=["text-embedding-3-small-aiberm"],
    )
    parser.add_argument("--ranks", nargs="+", type=int, default=[8, 16, 32, 64, 128])
    parser.add_argument(
        "--lambdas",
        nargs="+",
        type=float,
        default=[0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0],
    )
    parser.add_argument("--hard-negatives", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--oracle-steps", type=int, default=100)
    parser.add_argument("--oracle-learning-rate", type=float, default=0.05)
    parser.add_argument("--oracle-penalty", type=float, default=1e-3)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--random-state", type=int, default=0)
    return parser


if __name__ == "__main__":
    load_dotenv(ROOT / ".env")
    run(build_parser().parse_args())
