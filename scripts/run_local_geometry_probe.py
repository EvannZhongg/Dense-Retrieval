"""Probe whether corpus-local geometry predicts oracle correction coordinates.

For every leave-one-corpus-out fold, the correction basis, ridge predictors,
feature normalization, and ridge penalty are fitted on the remaining corpora.
The held-out corpus contributes only document embeddings to its unsupervised
prototype geometry. Its qrels are used once to score prediction on test queries.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dense_retrieval.analysis.local_geometry import (  # noqa: E402
    CorpusPrototypeGeometry,
    fit_reference_geometry,
    query_correction_features,
)
from dense_retrieval.analysis.query_correction import (  # noqa: E402
    build_oracle_deltas,
    corrected_queries,
    fit_zero_origin_pca_subspace,
)
from dense_retrieval.analysis.study_data import (  # noqa: E402
    DEFAULT_LAMBDAS,
    prepare_three_way,
)
from dense_retrieval.embeddings import MODEL_SPECS  # noqa: E402
from dense_retrieval.evaluation.ranking import evaluate_retrieval  # noqa: E402


METHODS = ("q_only", "reference_geometry", "corpus_geometry")


def _select_lambda(curve, rule):
    """Select a step from per-corpus dev metrics, with exact abstention."""
    if not curve or not any(abs(row["lambda"]) <= 1e-12 for row in curve):
        raise ValueError("lambda curve must include zero")
    if rule == "macro":
        return max(
            curve,
            key=lambda row: (
                row["macro_HitRate@10"],
                -row["lambda"],
            ),
        )["lambda"]
    if rule != "robust":
        raise ValueError("lambda selection rule must be macro or robust")
    baseline = next(row for row in curve if abs(row["lambda"]) <= 1e-12)
    eligible = [
        row
        for row in curve
        if row["lambda"] > 0
        and row["min_HitRate@10_gain"] >= -1e-12
        and row["macro_HitRate@10"] > baseline["macro_HitRate@10"] + 1e-12
    ]
    if not eligible:
        return 0.0
    return max(
        eligible,
        key=lambda row: (
            row["macro_HitRate@10"],
            row["macro_NDCG@10"],
            -row["lambda"],
        ),
    )["lambda"]


def _balanced_sample_weights(corpus_sizes: list[int]) -> np.ndarray:
    weights = [np.full(size, 1.0 / (len(corpus_sizes) * size)) for size in corpus_sizes]
    return np.concatenate(weights)


def _fit_normalizer(values: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.sum(values * weights[:, None], axis=0)
    variance = np.sum(weights[:, None] * (values - center) ** 2, axis=0)
    scale = np.sqrt(np.maximum(variance, 1e-12))
    scale = np.where(scale > 1e-6, scale, 1.0)
    return center.astype(np.float32), scale.astype(np.float32)


def _normalize(values: np.ndarray, center: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return np.clip((values - center) / scale, -8.0, 8.0).astype(np.float32)


def _target_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    squared_error = np.mean((target - prediction) ** 2)
    target_energy = np.mean(target**2)
    denominator = np.maximum(
        np.linalg.norm(target, axis=1) * np.linalg.norm(prediction, axis=1), 1e-12
    )
    cosine = np.sum(target * prediction, axis=1) / denominator
    return {
        "mse": float(squared_error),
        "normalized_mse": float(squared_error / max(target_energy, 1e-12)),
        "r2_variance_weighted": float(
            r2_score(target, prediction, multioutput="variance_weighted")
        ),
        "cosine_mean": float(np.mean(cosine)),
        "cosine_median": float(np.median(cosine)),
    }


def _split_target(dataset, queries, documents, components):
    rows, deltas, missing = build_oracle_deltas(dataset, queries, documents)
    return rows, (deltas @ components.T).astype(np.float32), missing


def _geometry_path(output_dir, model_key, corpus_name, args):
    return output_dir / "models" / (
        f"{model_key}_{corpus_name}_k{args.n_prototypes}_seed{args.random_state}.npz"
    )


def _load_or_fit_geometry(output_dir, model_key, corpus_name, documents, args):
    path = _geometry_path(output_dir, model_key, corpus_name, args)
    if path.exists():
        with np.load(path) as values:
            if (
                int(values["max_fit_documents"][0]) == args.max_fit_documents
                and int(values["max_iter"][0]) == args.prototype_max_iter
            ):
                return CorpusPrototypeGeometry(
                    values["prototypes"], values["occupancy"]
                )
    geometry = CorpusPrototypeGeometry.fit(
        documents,
        n_prototypes=args.n_prototypes,
        max_fit_documents=args.max_fit_documents,
        max_iter=args.prototype_max_iter,
        random_state=args.random_state,
        assignment_batch_size=args.assignment_batch_size,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        prototypes=geometry.prototypes.astype(np.float32),
        occupancy=geometry.occupancy.astype(np.float32),
        max_fit_documents=np.asarray([args.max_fit_documents]),
        max_iter=np.asarray([args.prototype_max_iter]),
    )
    return geometry


def _feature_matrix(queries, method, local_geometry, reference_geometry, components, args):
    return query_correction_features(
        queries,
        method,
        local_geometry,
        reference_geometry,
        components,
        top_m=args.top_m,
        temperature=args.temperature,
    )


def run(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_rows = []
    retrieval_rows = []
    selection_rows = []
    lambda_selection_rows = []
    metadata = []
    for model_key in args.models:
        loaded = {}
        local_geometries = {}
        for name in args.datasets:
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
            ) = prepare_three_way(name, model_key, args.cache_root)
            train_rows, train_deltas, train_missing = build_oracle_deltas(
                train_dataset, train_queries, documents
            )
            loaded[name] = {
                "train_dataset": train_dataset,
                "dev_dataset": dev_dataset,
                "test_dataset": test_dataset,
                "train_queries": train_queries,
                "dev_queries": dev_queries,
                "test_queries": test_queries,
                "documents": documents,
                "train_rows": train_rows,
                "train_deltas": train_deltas,
                "train_missing": train_missing,
                "cache_dir": str(cache_dir),
                "split_method": split_method,
            }
            local_geometries[name] = _load_or_fit_geometry(
                args.output_dir, model_key, name, documents, args
            )

        for fold_index, heldout in enumerate(args.datasets):
            train_names = [name for name in args.datasets if name != heldout]
            pooled_deltas = np.concatenate(
                [loaded[name]["train_deltas"] for name in train_names]
            )
            _mean, components = fit_zero_origin_pca_subspace(
                pooled_deltas, args.rank
            )
            reference_geometry = fit_reference_geometry(
                {name: loaded[name]["documents"] for name in train_names},
                n_prototypes=args.n_prototypes,
                max_fit_documents=args.max_fit_documents,
                max_iter=args.prototype_max_iter,
                random_state=args.random_state + 1009 * (fold_index + 1),
                assignment_batch_size=args.assignment_batch_size,
            )

            train_targets = {}
            dev_targets = {}
            dev_rows = {}
            for name in train_names:
                data = loaded[name]
                train_targets[name] = (
                    data["train_rows"],
                    (data["train_deltas"] @ components.T).astype(np.float32),
                )
                rows, targets, _missing = _split_target(
                    data["dev_dataset"],
                    data["dev_queries"],
                    data["documents"],
                    components,
                )
                dev_rows[name] = rows
                dev_targets[name] = targets

            held_data = loaded[heldout]
            held_rows, held_targets, held_missing = _split_target(
                held_data["test_dataset"],
                held_data["test_queries"],
                held_data["documents"],
                components,
            )
            baseline_metrics = evaluate_retrieval(
                held_data["test_dataset"],
                held_data["test_queries"],
                held_data["documents"],
                args.batch_size,
            )
            retrieval_rows.append(
                {
                    "model": model_key,
                    "heldout_corpus": heldout,
                    "method": "baseline",
                    "selected_lambda": 0.0,
                    **baseline_metrics,
                }
            )

            for method in METHODS:
                train_chunks = []
                target_chunks = []
                corpus_sizes = []
                for name in train_names:
                    rows, targets = train_targets[name]
                    features = _feature_matrix(
                        loaded[name]["train_queries"],
                        method,
                        local_geometries[name],
                        reference_geometry,
                        components,
                        args,
                    )[rows]
                    train_chunks.append(features)
                    target_chunks.append(targets)
                    corpus_sizes.append(len(rows))
                train_values = np.concatenate(train_chunks)
                train_target = np.concatenate(target_chunks)
                sample_weights = _balanced_sample_weights(corpus_sizes)
                center, scale = _fit_normalizer(train_values, sample_weights)
                normalized_train = _normalize(train_values, center, scale)

                best = None
                best_model = None
                for alpha in args.alphas:
                    model = Ridge(alpha=alpha)
                    model.fit(
                        normalized_train,
                        train_target,
                        sample_weight=sample_weights * len(sample_weights),
                    )
                    corpus_losses = []
                    for name in train_names:
                        features = _feature_matrix(
                            loaded[name]["dev_queries"],
                            method,
                            local_geometries[name],
                            reference_geometry,
                            components,
                            args,
                        )[dev_rows[name]]
                        prediction = model.predict(_normalize(features, center, scale))
                        corpus_losses.append(
                            _target_metrics(dev_targets[name], prediction)[
                                "normalized_mse"
                            ]
                        )
                    macro_loss = float(np.mean(corpus_losses))
                    selection_rows.append(
                        {
                            "model": model_key,
                            "heldout_corpus": heldout,
                            "method": method,
                            "alpha": float(alpha),
                            "macro_dev_normalized_mse": macro_loss,
                        }
                    )
                    candidate = (macro_loss, -float(alpha))
                    if best is None or candidate < best:
                        best = candidate
                        best_model = model

                per_lambda_metrics = []
                for lambda_ in args.lambdas:
                    corpus_metrics = {}
                    for name in train_names:
                        dev_features = _feature_matrix(
                            loaded[name]["dev_queries"],
                            method,
                            local_geometries[name],
                            reference_geometry,
                            components,
                            args,
                        )
                        dev_coordinates = best_model.predict(
                            _normalize(dev_features, center, scale)
                        )
                        dev_deltas = dev_coordinates @ components
                        metrics = evaluate_retrieval(
                            loaded[name]["dev_dataset"],
                            corrected_queries(
                                loaded[name]["dev_queries"], dev_deltas, lambda_
                            ),
                            loaded[name]["documents"],
                            args.batch_size,
                        )
                        corpus_metrics[name] = metrics
                        lambda_selection_rows.append(
                            {
                                "model": model_key,
                                "heldout_corpus": heldout,
                                "method": method,
                                "scope": name,
                                "lambda": float(lambda_),
                                "dev_HitRate@10": metrics["HitRate@10"],
                                "dev_NDCG@10": metrics["NDCG@10"],
                                "min_HitRate@10_gain": np.nan,
                            }
                        )
                    per_lambda_metrics.append(
                        {
                            "lambda": float(lambda_),
                            "metrics": corpus_metrics,
                            "macro_HitRate@10": float(
                                np.mean(
                                    [value["HitRate@10"] for value in corpus_metrics.values()]
                                )
                            ),
                            "macro_NDCG@10": float(
                                np.mean(
                                    [value["NDCG@10"] for value in corpus_metrics.values()]
                                )
                            ),
                        }
                    )
                zero = next(
                    row for row in per_lambda_metrics if abs(row["lambda"]) <= 1e-12
                )
                for row in per_lambda_metrics:
                    row["min_HitRate@10_gain"] = min(
                        row["metrics"][name]["HitRate@10"]
                        - zero["metrics"][name]["HitRate@10"]
                        for name in train_names
                    )
                    lambda_selection_rows.append(
                        {
                            "model": model_key,
                            "heldout_corpus": heldout,
                            "method": method,
                            "scope": "macro",
                            "lambda": row["lambda"],
                            "dev_HitRate@10": row["macro_HitRate@10"],
                            "dev_NDCG@10": row["macro_NDCG@10"],
                            "min_HitRate@10_gain": row["min_HitRate@10_gain"],
                        }
                    )
                selected_lambda = _select_lambda(
                    per_lambda_metrics, args.lambda_selection
                )

                test_features = _feature_matrix(
                    held_data["test_queries"],
                    method,
                    local_geometries[heldout],
                    reference_geometry,
                    components,
                    args,
                )
                prediction_all = best_model.predict(
                    _normalize(test_features, center, scale)
                )
                prediction = prediction_all[held_rows]
                result_rows.append(
                    {
                        "model": model_key,
                        "heldout_corpus": heldout,
                        "method": method,
                        "selected_alpha": float(-best[1]),
                        "test_queries": len(held_rows),
                        **_target_metrics(held_targets, prediction),
                    }
                )
                retrieval_metrics = evaluate_retrieval(
                    held_data["test_dataset"],
                    corrected_queries(
                        held_data["test_queries"],
                        prediction_all @ components,
                        selected_lambda,
                    ),
                    held_data["documents"],
                    args.batch_size,
                )
                retrieval_rows.append(
                    {
                        "model": model_key,
                        "heldout_corpus": heldout,
                        "method": method,
                        "selected_lambda": selected_lambda,
                        **retrieval_metrics,
                    }
                )

            metadata.append(
                {
                    "model": model_key,
                    "heldout_corpus": heldout,
                    "training_corpora": train_names,
                    "rank": len(components),
                    "n_prototypes": args.n_prototypes,
                    "top_m": args.top_m,
                    "temperature": args.temperature,
                    "max_fit_documents": args.max_fit_documents,
                    "prototype_max_iter": args.prototype_max_iter,
                    "random_state": args.random_state,
                    "prototype_fit_is_document_only": True,
                    "reference_geometry_uses_training_corpora_only": True,
                    "heldout_qrels_used_for_training_or_selection": False,
                    "heldout_test_missing_positive": held_missing,
                    "alpha_grid": list(args.alphas),
                    "lambda_grid": list(args.lambdas),
                    "lambda_selection": args.lambda_selection,
                }
            )
            print(f"{model_key}/{heldout}: complete", flush=True)

    result_frame = pd.DataFrame(result_rows)
    indexed = result_frame.set_index(["model", "heldout_corpus", "method"])
    q_only_nmse = indexed["normalized_mse"].xs("q_only", level="method")
    reference_nmse = indexed["normalized_mse"].xs(
        "reference_geometry", level="method"
    )
    result_frame["nmse_gain_over_q_only"] = [
        float(q_only_nmse.loc[(row.model, row.heldout_corpus)] - row.normalized_mse)
        for row in result_frame.itertuples()
    ]
    result_frame["nmse_gain_over_reference"] = [
        float(
            reference_nmse.loc[(row.model, row.heldout_corpus)] - row.normalized_mse
        )
        for row in result_frame.itertuples()
    ]
    result_frame.to_csv(
        args.output_dir / "local_geometry_loco_metrics.csv", index=False
    )
    pd.DataFrame(selection_rows).to_csv(
        args.output_dir / "local_geometry_alpha_selection.csv", index=False
    )
    pd.DataFrame(retrieval_rows).to_csv(
        args.output_dir / "local_geometry_loco_retrieval.csv", index=False
    )
    pd.DataFrame(lambda_selection_rows).to_csv(
        args.output_dir / "local_geometry_lambda_selection.csv", index=False
    )
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=ROOT / "cache")
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "results" / "local_geometry_probe"
    )
    parser.add_argument(
        "--datasets", nargs="+", default=["fiqa", "arguana", "scifact", "nfcorpus"]
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_SPECS,
        default=["text-embedding-3-small-aiberm"],
    )
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--n-prototypes", type=int, default=32)
    parser.add_argument("--top-m", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--max-fit-documents", type=int, default=20_000)
    parser.add_argument("--prototype-max-iter", type=int, default=30)
    parser.add_argument("--assignment-batch-size", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--alphas",
        nargs="+",
        type=float,
        default=[10.0, 100.0, 1000.0, 10_000.0, 100_000.0],
    )
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument(
        "--lambdas", nargs="+", type=float, default=DEFAULT_LAMBDAS
    )
    parser.add_argument(
        "--lambda-selection", choices=["macro", "robust"], default="macro"
    )
    return parser


if __name__ == "__main__":
    load_dotenv(ROOT / ".env")
    run(build_parser().parse_args())
