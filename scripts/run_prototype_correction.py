"""Train and evaluate a leakage-free Prototype Correction Field.

Document prototypes are fitted once from frozen corpus embeddings.  Only the
training queries' oracle correction coordinates are used to learn prototype
values; test qrels are used solely for evaluation by the existing oracle
pipeline.
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

from dense_retrieval.analysis.prototype_correction import (  # noqa: E402
    PrototypeCorrectionField,
    fit_spherical_kmeans,
    fit_prototype_values,
    predict_prototype_values,
    prototype_weights,
    project_correction_targets,
)
from dense_retrieval.analysis.query_correction import (  # noqa: E402
    build_oracle_deltas,
    fit_pca_subspace,
    prepare_correction_data,
)
from dense_retrieval.embeddings import MODEL_SPECS  # noqa: E402
from dense_retrieval.evaluation.ranking import evaluate_retrieval  # noqa: E402
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




def run(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = args.output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    metadata = []

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
            ) = prepare_correction_data(
                dataset_name,
                model_spec,
                cache_root=args.cache_root,
                datasets_root=ROOT / "datasets",
                configs_root=ROOT / "configs",
                split_salt=f"prototype-field-v1:{dataset_name}",
                missing_relevant_policy="keep" if dataset_name == "arguana" else "error",
            )
            train_rows, train_deltas, train_missing = build_oracle_deltas(
                train_dataset, train_queries, documents
            )
            # B is learned from train deltas only. It is not needed to fit the
            # field itself, but allows the saved V to be mapped back to deltas.
            mean, components = fit_pca_subspace(
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
            baseline = evaluate_retrieval(
                test_dataset, test_queries, documents, args.batch_size
            )
            for lambda_ in args.lambdas:
                corrected = field.apply(test_queries, lambda_=lambda_)
                metrics = evaluate_retrieval(
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
        "--models", nargs="+", choices=MODEL_SPECS, default=list(MODEL_SPECS)
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