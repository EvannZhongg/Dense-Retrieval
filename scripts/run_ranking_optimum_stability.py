"""Test whether privileged ranking-oracle coordinates are identifiable.

The audit holds the query, frozen corpus, and one LOCO-fitted rank-32 basis
fixed. It independently re-optimizes each held-out query while perturbing one
optimization choice at a time. Held-out qrels are used only for this privileged
diagnostic; no predictor is fitted and no deployable parameter is selected.
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
    optimize_ranking_oracle_solution,
    ranking_boundary_directions,
)
from dense_retrieval.analysis.oracle_stability import (  # noqa: E402
    pairwise_solution_stability,
    per_query_ranking_loss,
    per_query_retrieval_metrics,
    subsample_rank_records,
    summarize_pairwise_stability,
)
from dense_retrieval.analysis.ranking_correction import build_rank_records  # noqa: E402
from dense_retrieval.analysis.study_data import prepare_three_way  # noqa: E402
from dense_retrieval.embeddings import MODEL_SPECS  # noqa: E402
from dense_retrieval.retrieval.exact import clear_cuda_document_cache  # noqa: E402


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
    _, train_directions = ranking_boundary_directions(
        train_dataset,
        train_queries,
        documents,
        hard_negatives=args.reference_negative_count,
        temperature=args.temperature,
    )
    return {
        "train_directions": train_directions,
        "test_dataset": test_dataset,
        "test_queries": test_queries,
        "documents": documents,
        "metadata": {
            "split": split,
            "cache_dir": str(cache_dir.relative_to(ROOT)),
            "train_boundary_directions": len(train_directions),
            "test_queries": len(test_dataset.queries),
        },
    }


def _run_specs(args: argparse.Namespace) -> list[dict]:
    specs = [
        {
            "run_id": "reference",
            "factor": "reference",
            "factor_value": "reference",
            "optimizer_seed": 0,
            "initialization_seed": None,
            "negative_sample_seed": None,
            "negative_count": args.reference_negative_count,
            "penalty": args.reference_penalty,
        }
    ]
    for seed in range(1, args.repeats + 1):
        specs.extend(
            [
                {
                    "run_id": f"initialization_{seed}",
                    "factor": "initialization",
                    "factor_value": str(seed),
                    "optimizer_seed": 0,
                    "initialization_seed": seed,
                    "negative_sample_seed": None,
                    "negative_count": args.reference_negative_count,
                    "penalty": args.reference_penalty,
                },
                {
                    "run_id": f"negative_sample_{seed}",
                    "factor": "negative_sample",
                    "factor_value": str(seed),
                    "optimizer_seed": 0,
                    "initialization_seed": None,
                    "negative_sample_seed": seed,
                    "negative_count": args.reference_negative_count,
                    "penalty": args.reference_penalty,
                },
                {
                    "run_id": f"optimizer_seed_{seed}",
                    "factor": "optimizer_seed",
                    "factor_value": str(seed),
                    "optimizer_seed": seed,
                    "initialization_seed": None,
                    "negative_sample_seed": None,
                    "negative_count": args.reference_negative_count,
                    "penalty": args.reference_penalty,
                },
            ]
        )
    for count in args.negative_counts:
        if count != args.reference_negative_count:
            specs.append(
                {
                    "run_id": f"negative_count_{count}",
                    "factor": "negative_count",
                    "factor_value": str(count),
                    "optimizer_seed": 0,
                    "initialization_seed": None,
                    "negative_sample_seed": None,
                    "negative_count": count,
                    "penalty": args.reference_penalty,
                }
            )
    for penalty in args.penalties:
        if not np.isclose(penalty, args.reference_penalty):
            specs.append(
                {
                    "run_id": f"regularization_{penalty:g}",
                    "factor": "regularization",
                    "factor_value": f"{penalty:g}",
                    "optimizer_seed": 0,
                    "initialization_seed": None,
                    "negative_sample_seed": None,
                    "negative_count": args.reference_negative_count,
                    "penalty": penalty,
                }
            )
    return specs


def _write_findings(output_dir: Path, summary: pd.DataFrame) -> None:
    pooled = (
        summary.groupby("comparison_group", as_index=False)
        .agg(
            coordinate_cosine=("mean_coordinate_cosine", "mean"),
            coordinate_cosine_p10=("p10_coordinate_cosine", "mean"),
            top10_overlap=("mean_top_k_overlap", "mean"),
            abs_ndcg_difference=("mean_abs_ndcg_difference", "mean"),
            abs_mrr_difference=("mean_abs_mrr_difference", "mean"),
            abs_reference_rank_loss_difference=(
                "mean_abs_reference_rank_loss_difference",
                "mean",
            ),
            directionally_distinct_fraction=(
                "directionally_distinct_fraction",
                "mean",
            ),
            distinct_but_metric_equivalent_fraction=(
                "directionally_distinct_but_metric_equivalent_fraction",
                "mean",
            ),
            distinct_but_ranking_equivalent_fraction=(
                "directionally_distinct_but_ranking_equivalent_fraction",
                "mean",
            ),
        )
        .sort_values("comparison_group")
    )
    columns = list(pooled.columns)
    table_rows = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for values in pooled.itertuples(index=False, name=None):
        table_rows.append(
            "| "
            + " | ".join(
                f"{value:.4f}" if isinstance(value, float) else str(value)
                for value in values
            )
            + " |"
        )
    table = "\n".join(table_rows)
    (output_dir / "findings.md").write_text(
        "# Ranking-optimum stability audit\n\n"
        "This is a privileged identifiability diagnostic, not a deployable "
        "correction result. A high retrieval overlap paired with low coordinate "
        "cosine is direct evidence that coordinate regression has no unique "
        "target under the current ranking objective. Directionally distinct "
        "means cosine below 0.2; metric equivalence requires NDCG@10 and MRR@10 "
        "differences at most 0.01, and ranking equivalence additionally requires "
        "Top-10 overlap of at least 0.8.\n\n"
        f"{table}\n",
        encoding="utf-8",
    )


def run(args: argparse.Namespace) -> None:
    if args.rank < 1 or args.repeats < 1:
        raise ValueError("rank and repeats must be positive")
    if args.reference_negative_count > args.hard_negative_pool:
        raise ValueError("reference negative count exceeds hard-negative pool")
    if max(args.negative_counts) > args.hard_negative_pool:
        raise ValueError("negative count exceeds hard-negative pool")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_metric_rows = []
    pairwise_frames = []
    fold_metadata = []
    specs = _run_specs(args)

    for model_key in args.models:
        corpora = {
            name: _prepare_corpus(name, model_key, args) for name in args.datasets
        }
        model_label = MODEL_SPECS[model_key]["label"]
        for heldout_name in args.datasets:
            training_names = [name for name in args.datasets if name != heldout_name]
            directions = np.concatenate(
                [corpora[name]["train_directions"] for name in training_names], axis=0
            )
            basis = fit_svd_subspace(directions, args.rank)
            heldout = corpora[heldout_name]
            pool_records = build_rank_records(
                heldout["test_dataset"],
                heldout["test_queries"],
                heldout["documents"],
                args.hard_negative_pool,
            )
            solutions_by_id = {}
            for spec in specs:
                records = subsample_rank_records(
                    pool_records,
                    spec["negative_count"],
                    random_state=spec["negative_sample_seed"],
                )
                initial = None
                if spec["initialization_seed"] is not None:
                    rng = np.random.default_rng(spec["initialization_seed"])
                    initial = rng.normal(
                        0.0,
                        args.initialization_scale,
                        size=(len(records["query_rows"]), args.rank),
                    ).astype(np.float32)
                solution = optimize_ranking_oracle_solution(
                    heldout["test_queries"],
                    heldout["documents"],
                    records,
                    basis,
                    steps=args.oracle_steps,
                    learning_rate=args.oracle_learning_rate,
                    penalty=spec["penalty"],
                    temperature=args.temperature,
                    device=args.device,
                    initial_coordinates=initial,
                    optimizer_seed=spec["optimizer_seed"],
                )
                source_queries = heldout["test_queries"][solution.query_rows]
                corrected = source_queries + solution.deltas
                corrected /= np.maximum(
                    np.linalg.norm(corrected, axis=1, keepdims=True), 1e-12
                )
                metrics, rankings = per_query_retrieval_metrics(
                    heldout["test_dataset"],
                    corrected,
                    heldout["documents"],
                    records,
                    top_k=10,
                )
                metrics["reference_rank_loss"] = per_query_ranking_loss(
                    corrected,
                    heldout["documents"],
                    subsample_rank_records(
                        pool_records, args.reference_negative_count
                    ),
                    temperature=args.temperature,
                )
                query_ids = metrics["query_id"].astype(str).tolist()
                solutions_by_id[spec["run_id"]] = {
                    **spec,
                    "query_rows": solution.query_rows,
                    "query_ids": query_ids,
                    "coordinates": solution.coordinates,
                    "deltas": solution.deltas,
                    "metrics": metrics,
                    "rankings": rankings,
                    "rank_loss": solution.rank_loss,
                }
                enriched = metrics.assign(
                    heldout_corpus=heldout_name,
                    model=model_label,
                    run_id=spec["run_id"],
                    factor=spec["factor"],
                    factor_value=spec["factor_value"],
                    rank_loss=solution.rank_loss,
                    mean_coordinate_norm=float(
                        np.mean(np.linalg.norm(solution.coordinates, axis=1))
                    ),
                    mean_displacement_norm=float(
                        np.mean(np.linalg.norm(solution.deltas, axis=1))
                    ),
                )
                run_metric_rows.append(enriched)
                print(
                    f"{model_key}/{heldout_name}/{spec['run_id']}: complete",
                    flush=True,
                )

            comparison_solutions = []
            factors = sorted(
                {spec["factor"] for spec in specs if spec["factor"] != "reference"}
            )
            for factor in factors:
                comparison_solutions.append(
                    {**solutions_by_id["reference"], "comparison_group": factor}
                )
                comparison_solutions.extend(
                    {
                        **solutions_by_id[spec["run_id"]],
                        "comparison_group": factor,
                    }
                    for spec in specs
                    if spec["factor"] == factor
                )
            pairwise = pairwise_solution_stability(comparison_solutions)
            pairwise.insert(0, "model", model_label)
            pairwise.insert(0, "heldout_corpus", heldout_name)
            pairwise_frames.append(pairwise)
            fold_metadata.append(
                {
                    "model": model_label,
                    "heldout_corpus": heldout_name,
                    "training_corpora": training_names,
                    "pooled_train_boundary_directions": len(directions),
                    "optimized_test_queries": len(pool_records["query_rows"]),
                    "heldout": heldout["metadata"],
                }
            )
        clear_cuda_document_cache()

    run_metrics = pd.concat(run_metric_rows, ignore_index=True)
    pairwise = pd.concat(pairwise_frames, ignore_index=True)
    summary = summarize_pairwise_stability(pairwise)
    run_metrics.to_csv(args.output_dir / "run_query_metrics.csv", index=False)
    pairwise.to_csv(args.output_dir / "pairwise_query_stability.csv", index=False)
    summary.to_csv(args.output_dir / "stability_summary.csv", index=False)
    (args.output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "protocol": "LOCO privileged ranking-optimum identifiability audit",
                "predictor_trained": False,
                "heldout_qrels_used_for_basis_fit": False,
                "heldout_qrels_used_for_oracle_diagnostic": True,
                "basis": "ranking_boundary_svd",
                "rank": args.rank,
                "run_specs": specs,
                "oracle_steps": args.oracle_steps,
                "oracle_learning_rate": args.oracle_learning_rate,
                "temperature": args.temperature,
                "hard_negative_pool": args.hard_negative_pool,
                "initialization_scale": args.initialization_scale,
                "folds": fold_metadata,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_findings(args.output_dir, summary)
    print(summary.round(4).to_string(index=False), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=ROOT / "cache")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "ranking_optimum_stability",
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
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--hard-negative-pool", type=int, default=128)
    parser.add_argument("--reference-negative-count", type=int, default=32)
    parser.add_argument(
        "--negative-counts", nargs="+", type=int, default=[8, 16, 32, 64]
    )
    parser.add_argument(
        "--penalties", nargs="+", type=float, default=[0.0, 1e-4, 1e-3, 1e-2]
    )
    parser.add_argument("--reference-penalty", type=float, default=1e-3)
    parser.add_argument("--initialization-scale", type=float, default=0.05)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--oracle-steps", type=int, default=100)
    parser.add_argument("--oracle-learning-rate", type=float, default=0.05)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser


if __name__ == "__main__":
    load_dotenv(ROOT / ".env")
    run(build_parser().parse_args())
