"""Compare query-only and corpus-conditioned correction fields.

Every method in this study sees the same train/dev/test query split.  Model
parameters are fitted on train queries only; dev selects the correction
strength and the selected strength is evaluated once on test queries.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dense_retrieval.analysis.prototype_correction import (  # noqa: E402
    PrototypeCorrectionField,
    project_correction_targets,
)
from dense_retrieval.analysis.query_correction import (  # noqa: E402
    build_oracle_deltas,
    corrected_queries,
    fit_pca_subspace,
)
from dense_retrieval.analysis.study_data import (  # noqa: E402
    DEFAULT_LAMBDAS,
    prepare_three_way,
)
from dense_retrieval.embeddings import MODEL_SPECS  # noqa: E402
from dense_retrieval.evaluation.ranking import METRICS, evaluate_retrieval  # noqa: E402


METHODS = ["baseline", "mean_correction", "linear_q_only", "mlp_q_only", "prototype_field"]
RIDGE_ALPHAS = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]


def stable_train_validation_split(query_ids, validation_fraction: float = 0.2):
    ordered = sorted(
        query_ids,
        key=lambda query_id: hashlib.sha256(f"ablation-model-selection-v1:{query_id}".encode()).digest(),
    )
    count = max(1, int(round(len(ordered) * validation_fraction)))
    validation_ids = set(ordered[:count])
    validation = np.asarray([query_id in validation_ids for query_id in query_ids])
    return ~validation, validation


def fit_linear_q_only(x, targets, query_ids):
    fit_mask, validation_mask = stable_train_validation_split(query_ids)
    candidates = []
    for alpha in RIDGE_ALPHAS:
        model = Ridge(alpha=alpha, fit_intercept=True)
        model.fit(x[fit_mask], targets[fit_mask])
        candidates.append((mean_squared_error(targets[validation_mask], model.predict(x[validation_mask])), alpha))
    alpha = min(candidates)[1]
    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(x, targets)
    return model, float(alpha)


def fit_mlp_q_only(x, targets, random_state: int = 0):
    model = MLPRegressor(
        hidden_layer_sizes=(256,),
        activation="relu",
        solver="adam",
        batch_size=128,
        learning_rate_init=1e-3,
        max_iter=200,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
        random_state=random_state,
        verbose=False,
    )
    model.fit(x, targets)
    return model


def select_lambda(dev_dataset, dev_queries, documents, predicted_deltas, lambdas, batch_size):
    rows = []
    for lambda_ in lambdas:
        metrics = evaluate_retrieval(
            dev_dataset,
            corrected_queries(dev_queries, predicted_deltas, lambda_),
            documents,
            batch_size,
        )
        rows.append({"lambda": float(lambda_), **metrics})
    frame = pd.DataFrame(rows)
    best = frame.loc[frame["HitRate@10"].idxmax()]
    return float(best["lambda"]), frame


def run(args: argparse.Namespace):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    curve_rows = []
    metadata = []
    for dataset_name in args.datasets:
        for model_key in args.models:
            (
                train_dataset,
                dev_dataset,
                test_dataset,
                train_queries,
                dev_queries,
                test_queries,
                documents,
                cache_dir,
                split_method,
            ) = prepare_three_way(dataset_name, model_key, args.cache_root)
            train_rows, train_deltas, train_missing = build_oracle_deltas(
                train_dataset, train_queries, documents
            )
            mean, components = fit_pca_subspace(train_deltas, args.rank)
            train_coordinates = project_correction_targets(train_deltas, mean, components)
            query_ids = [train_dataset.queries[row].query_id for row in train_rows]

            field = PrototypeCorrectionField.fit(
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
            linear, linear_alpha = fit_linear_q_only(
                train_queries[train_rows], train_coordinates, query_ids
            )
            mlp = fit_mlp_q_only(
                train_queries[train_rows], train_coordinates, args.random_state
            )
            models = {
                "mean_correction": lambda q: np.broadcast_to(mean, (len(q), len(mean))),
                "linear_q_only": lambda q: mean + linear.predict(q) @ components,
                "mlp_q_only": lambda q: mean + mlp.predict(q) @ components,
                "prototype_field": lambda q: field.predict_deltas(q),
            }
            baseline_dev = evaluate_retrieval(dev_dataset, dev_queries, documents, args.batch_size)
            baseline_test = evaluate_retrieval(test_dataset, test_queries, documents, args.batch_size)
            summary_rows.append(
                {"dataset": dataset_name, "model": model_key, "method": "baseline", "selected_lambda": 0.0, **baseline_test}
            )
            curve_rows.append(
                {"dataset": dataset_name, "model": model_key, "method": "baseline", "lambda": 0.0, **baseline_dev}
            )
            selected = {"baseline": 0.0}
            for method, predictor in models.items():
                dev_deltas = predictor(dev_queries)
                selected_lambda, curve = select_lambda(
                    dev_dataset, dev_queries, documents, dev_deltas, args.lambdas, args.batch_size
                )
                curve.insert(0, "method", method)
                curve.insert(0, "model", model_key)
                curve.insert(0, "dataset", dataset_name)
                curve_rows.extend(curve.to_dict("records"))
                test_deltas = predictor(test_queries)
                test_metrics = evaluate_retrieval(
                    test_dataset,
                    corrected_queries(test_queries, test_deltas, selected_lambda),
                    documents,
                    args.batch_size,
                )
                summary_rows.append(
                    {
                        "dataset": dataset_name,
                        "model": model_key,
                        "method": method,
                        "selected_lambda": selected_lambda,
                        **test_metrics,
                    }
                )
                selected[method] = selected_lambda
            metadata.append(
                {
                    "dataset": dataset_name,
                    "model": model_key,
                    "split_method": split_method,
                    "cache_dir": str(cache_dir.relative_to(ROOT)),
                    "train_queries": len(train_dataset.queries),
                    "dev_queries": len(dev_dataset.queries),
                    "test_queries": len(test_dataset.queries),
                    "train_queries_used": len(train_rows),
                    "train_missing_positive": train_missing,
                    "rank": args.rank,
                    "n_prototypes": args.n_prototypes,
                    "top_m": args.top_m,
                    "temperature": args.temperature,
                    "ridge": args.ridge,
                    "linear_alpha": linear_alpha,
                    "mlp_hidden_layer_sizes": [256],
                    "mlp_n_iter": int(mlp.n_iter_),
                    "selected_lambdas": selected,
                    "test_qrels_used_for_fitting": False,
                }
            )
            print(f"{dataset_name}/{model_key}: complete", flush=True)
    pd.DataFrame(summary_rows).to_csv(args.output_dir / "ablation_test_metrics.csv", index=False)
    pd.DataFrame(curve_rows).to_csv(args.output_dir / "ablation_dev_curves.csv", index=False)
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=ROOT / "cache")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "correction_ablation")
    parser.add_argument("--datasets", nargs="+", default=["fiqa", "arguana", "scifact", "nfcorpus"])
    parser.add_argument("--models", nargs="+", choices=MODEL_SPECS, default=list(MODEL_SPECS))
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--n-prototypes", type=int, default=64)
    parser.add_argument("--top-m", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--lambdas", nargs="+", type=float, default=DEFAULT_LAMBDAS)
    parser.add_argument("--batch-size", type=int, default=128)
    return parser


if __name__ == "__main__":
    load_dotenv(ROOT / ".env")
    run(build_parser().parse_args())
