"""Predict held-out low-rank oracle coefficients with a linear query model."""
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
from dotenv import load_dotenv
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dense_retrieval.analysis.query_correction import (  # noqa: E402
    apply_correction,
    build_oracle_deltas,
    gain_retention,
    load_model_config,
    prepare_correction_data,
    project_deltas,
)
from dense_retrieval.datasets import RetrievalDataset, load_beir_dataset  # noqa: E402
from dense_retrieval.datasets.splits import rows_for_queries  # noqa: E402
from dense_retrieval.embeddings import (  # noqa: E402
    MODEL_SPECS,
    cached_encode_queries,
    create_embedding_model,
)
from dense_retrieval.evaluation.ranking import (  # noqa: E402
    METRICS,
    evaluate_retrieval,
)
METHODS = [
    "baseline",
    "full_oracle",
    "subspace_oracle_r128",
    "predicted_subspace_r128",
]
COLORS = {
    "baseline": "#666666",
    "full_oracle": "#111111",
    "subspace_oracle_r128": "#c4513b",
    "predicted_subspace_r128": "#2f6f9f",
}


def subset_dataset(dataset, name, ids):
    ids = set(ids)
    queries = [sample for sample in dataset.queries if sample.query_id in ids]
    qrels = {sample.query_id: dataset.qrels[sample.query_id] for sample in queries}
    return RetrievalDataset(name, queries, dataset.corpus, qrels)


def split_arguana_dev_test(holdout):
    ordered = sorted(
        holdout.queries,
        key=lambda sample: hashlib.sha256(
            f"arguana-predictability-v1:{sample.query_id}".encode()
        ).digest(),
    )
    middle = len(ordered) // 2
    return (
        subset_dataset(holdout, "arguana_dev", [x.query_id for x in ordered[:middle]]),
        subset_dataset(holdout, "arguana_test", [x.query_id for x in ordered[middle:]]),
    )


def stable_regression_split(query_ids, validation_fraction=0.2):
    ordered = sorted(
        query_ids,
        key=lambda query_id: hashlib.sha256(
            f"coefficient-ridge-v1:{query_id}".encode()
        ).digest(),
    )
    validation_count = max(1, int(round(len(ordered) * validation_fraction)))
    validation_ids = set(ordered[:validation_count])
    validation = np.asarray([query_id in validation_ids for query_id in query_ids])
    return ~validation, validation


def select_and_fit_linear_model(x, y, query_ids, alphas):
    fit_mask, validation_mask = stable_regression_split(query_ids)
    validation_rows = []
    best = None
    for alpha in alphas:
        model = Ridge(alpha=float(alpha), fit_intercept=True)
        model.fit(x[fit_mask], y[fit_mask])
        prediction = model.predict(x[validation_mask])
        mse = mean_squared_error(y[validation_mask], prediction)
        validation_rows.append({"alpha": float(alpha), "validation_mse": float(mse)})
        candidate = (float(mse), float(alpha))
        if best is None or candidate < best:
            best = candidate
    model = Ridge(alpha=best[1], fit_intercept=True)
    model.fit(x, y)
    return model, pd.DataFrame(validation_rows)


def coefficient_metrics(train_target, train_prediction, test_target, test_prediction):
    numerator = np.sum(test_target * test_prediction, axis=1)
    denominator = np.linalg.norm(test_target, axis=1) * np.linalg.norm(
        test_prediction, axis=1
    )
    cosine = numerator / np.maximum(denominator, 1e-12)
    return {
        "train_mse": float(mean_squared_error(train_target, train_prediction)),
        "train_r2_variance_weighted": float(
            r2_score(train_target, train_prediction, multioutput="variance_weighted")
        ),
        "test_mse": float(mean_squared_error(test_target, test_prediction)),
        "test_r2_variance_weighted": float(
            r2_score(test_target, test_prediction, multioutput="variance_weighted")
        ),
        "test_coefficient_cosine_mean": float(np.mean(cosine)),
        "test_coefficient_cosine_median": float(np.median(cosine)),
        "test_relative_mse": float(
            np.sum((test_target - test_prediction) ** 2)
            / np.maximum(np.sum(test_target**2), 1e-12)
        ),
    }


def prepare_splits(dataset_name, cache_root, model_spec):
    (
        train_dataset,
        original_test_dataset,
        train_queries,
        original_test_queries,
        documents,
        cache_dir,
        _,
) = prepare_correction_data(
        dataset_name,
        model_spec,
        cache_root=cache_root,
        datasets_root=ROOT / "datasets",
        configs_root=ROOT / "configs",
        split_salt="arguana-low-rank-v1",
        missing_relevant_policy="keep",
    )

    if dataset_name == "fiqa":
        dev_dataset = load_beir_dataset("fiqa", ROOT / "datasets", "dev", False)
        dev_path = cache_dir / "queries_dev.npy"
        if dev_path.exists():
            dev_queries = np.load(dev_path, mmap_mode="r")
        else:
            embedding_model = create_embedding_model(
                load_model_config(ROOT / "configs", model_spec)["model"]
            )
            dev_queries = cached_encode_queries(
                embedding_model, dev_dataset.query_texts, dev_path
            )
        test_dataset = original_test_dataset
        test_queries = original_test_queries
        split_method = "official FiQA train/dev/test qrels"
    else:
        full_dataset = load_beir_dataset(
            "arguana", ROOT / "datasets", "test", False, "keep"
        )
        dev_dataset, test_dataset = split_arguana_dev_test(original_test_dataset)
        full_queries = np.load(cache_dir / "queries.npy", mmap_mode="r")
        dev_queries = full_queries[rows_for_queries(full_dataset, dev_dataset)]
        test_queries = full_queries[rows_for_queries(full_dataset, test_dataset)]
        split_method = "stable 70/15/15 split of official ArguAna test queries"

    return (
        train_dataset,
        dev_dataset,
        test_dataset,
        np.asarray(train_queries),
        np.asarray(dev_queries),
        np.asarray(test_queries),
        documents,
        cache_dir,
        split_method,
    )


def load_subspace(output_root, dataset, model_key, rank):
    path = output_root / "low_rank_oracle" / "subspaces" / f"{dataset}_{model_key}.npz"
    data = np.load(path)
    return data["mean"].astype(np.float64), data["components"][:rank].astype(np.float64)


def predicted_delta(queries, model, mean, components):
    coefficients = model.predict(np.asarray(queries, dtype=np.float64))
    return mean + coefficients @ components


def evaluate_methods(
    dataset,
    queries,
    documents,
    oracle_rows,
    full_deltas,
    subspace_deltas,
    predicted_deltas,
    lambdas,
    batch_size,
):
    rows = []
    baseline = evaluate_retrieval(dataset, queries, documents, batch_size)
    all_rows = np.arange(len(queries), dtype=np.int64)
    for lambda_ in lambdas:
        method_deltas = {
            "baseline": None,
            "full_oracle": (oracle_rows, full_deltas),
            "subspace_oracle_r128": (oracle_rows, subspace_deltas),
            "predicted_subspace_r128": (all_rows, predicted_deltas),
        }
        for method, correction in method_deltas.items():
            if method == "baseline":
                metrics = baseline
            else:
                corrected = apply_correction(
                    queries, correction[0], correction[1], lambda_
                )
                metrics = evaluate_retrieval(dataset, corrected, documents, batch_size)
            rows.append({"lambda": lambda_, "method": method, **metrics})
    return pd.DataFrame(rows)


def gain_retention_rows(summary):
    rows = []
    for (dataset, model, lambda_), group in summary.groupby(
        ["dataset", "model", "lambda"]
    ):
        values = group.set_index("method")
        for method in ("subspace_oracle_r128", "predicted_subspace_r128"):
            for metric in METRICS:
                baseline = values.loc["baseline", metric]
                full = values.loc["full_oracle", metric]
                subspace = values.loc["subspace_oracle_r128", metric]
                value = values.loc[method, metric]
                rows.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "lambda": lambda_,
                        "method": method,
                        "metric": metric,
                        "value": value,
                        "baseline": baseline,
                        "full_oracle": full,
                        "subspace_oracle": subspace,
                        "retention_vs_full_oracle": gain_retention(
                            value, baseline, full
                        ),
                        "retention_vs_subspace_oracle": gain_retention(
                            value, baseline, subspace
                        ),
                    }
                )
    return pd.DataFrame(rows)


def plot_retrieval_curves(summary, dataset, model, output_path):
    subset = summary[
        (summary["dataset"] == dataset) & (summary["model"] == model)
    ]
    figure, axes = plt.subplots(2, 3, figsize=(12, 7.2), sharex=True)
    for axis, metric in zip(axes.flat, METRICS):
        for method in METHODS:
            rows = subset[subset["method"] == method]
            axis.plot(
                rows["lambda"],
                rows[metric],
                marker="o",
                linewidth=2,
                linestyle="--" if method == "baseline" else "-",
                color=COLORS[method],
                label=method.replace("_", " "),
            )
        axis.set_title(metric)
        axis.set_xlabel(r"Correction $\lambda$")
        axis.set_ylim(0, 1.01)
        axis.grid(True, alpha=0.25)
    axes[0, 0].legend(frameon=False, fontsize=7)
    figure.suptitle(f"{dataset} / {model}: coefficient predictability")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_coefficient_metrics(frame, output_path):
    labels = [f"{row.dataset}\n{row.model}" for row in frame.itertuples()]
    positions = np.arange(len(frame))
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    axes[0].bar(positions, frame["test_r2_variance_weighted"], color="#2f6f9f")
    axes[0].axhline(0, color="#777777", linewidth=1)
    axes[0].set_title("Held-out coefficient R2")
    axes[1].bar(positions, frame["test_coefficient_cosine_mean"], color="#41844b")
    axes[1].set_title("Held-out coefficient cosine")
    for axis in axes:
        axis.set_xticks(positions, labels, rotation=30, ha="right")
        axis.grid(True, axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def run(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = args.output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    result_frames = []
    coefficient_rows = []
    validation_frames = []
    selected_rows = []
    metadata = []

    for dataset_name in args.datasets:
        for model_key in args.models:
            spec = MODEL_SPECS[model_key]
            model_label = spec["label"]
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
            ) = prepare_splits(dataset_name, args.cache_root, spec)
            mean, components = load_subspace(
                args.results_root, dataset_name, model_key, args.rank
            )

            train_rows, train_deltas, train_missing = build_oracle_deltas(
                train_dataset, train_queries, documents
            )
            dev_rows, dev_deltas, dev_missing = build_oracle_deltas(
                dev_dataset, dev_queries, documents
            )
            test_rows, test_deltas, test_missing = build_oracle_deltas(
                test_dataset, test_queries, documents
            )
            train_coefficients = (train_deltas - mean) @ components.T
            test_coefficients = (test_deltas - mean) @ components.T
            train_ids = [train_dataset.queries[row].query_id for row in train_rows]
            linear_model, validation = select_and_fit_linear_model(
                train_queries[train_rows],
                train_coefficients,
                train_ids,
                args.alphas,
            )
            validation.insert(0, "model", model_label)
            validation.insert(0, "dataset", dataset_name)
            validation_frames.append(validation)

            train_prediction = linear_model.predict(train_queries[train_rows])
            test_prediction = linear_model.predict(test_queries[test_rows])
            coefficient_rows.append(
                {
                    "dataset": dataset_name,
                    "model": model_label,
                    "rank": args.rank,
                    "selected_alpha": float(linear_model.alpha),
                    **coefficient_metrics(
                        train_coefficients,
                        train_prediction,
                        test_coefficients,
                        test_prediction,
                    ),
                }
            )
            np.savez_compressed(
                model_dir / f"{dataset_name}_{model_key}_r{args.rank}.npz",
                W=linear_model.coef_.T.astype(np.float32),
                b=linear_model.intercept_.astype(np.float32),
                alpha=np.asarray([linear_model.alpha], dtype=np.float32),
                mean=mean.astype(np.float32),
                components=components.astype(np.float32),
            )

            dev_subspace = project_deltas(
                dev_deltas, mean, components, args.rank
            )
            test_subspace = project_deltas(
                test_deltas, mean, components, args.rank
            )
            dev_predicted = predicted_delta(
                dev_queries, linear_model, mean, components
            )
            test_predicted = predicted_delta(
                test_queries, linear_model, mean, components
            )
            dev_summary = evaluate_methods(
                dev_dataset,
                dev_queries,
                documents,
                dev_rows,
                dev_deltas,
                dev_subspace,
                dev_predicted,
                args.lambdas,
                args.batch_size,
            )
            predicted_dev = dev_summary[
                dev_summary["method"] == "predicted_subspace_r128"
            ]
            best_hit = predicted_dev["HitRate@10"].max()
            selected_lambda = float(
                predicted_dev[predicted_dev["HitRate@10"] == best_hit]["lambda"].min()
            )

            test_summary = evaluate_methods(
                test_dataset,
                test_queries,
                documents,
                test_rows,
                test_deltas,
                test_subspace,
                test_predicted,
                args.lambdas,
                args.batch_size,
            )
            test_summary.insert(0, "model", model_label)
            test_summary.insert(0, "dataset", dataset_name)
            result_frames.append(test_summary)
            chosen = test_summary[np.isclose(test_summary["lambda"], selected_lambda)].copy()
            chosen.insert(2, "lambda_selection_metric", "dev HitRate@10")
            chosen.insert(3, "dev_selected_lambda", selected_lambda)
            selected_rows.append(chosen)

            train_id_set = {sample.query_id for sample in train_dataset.queries}
            dev_id_set = {sample.query_id for sample in dev_dataset.queries}
            test_id_set = {sample.query_id for sample in test_dataset.queries}
            if train_id_set & dev_id_set or train_id_set & test_id_set or dev_id_set & test_id_set:
                raise AssertionError("Train/dev/test query leakage")
            metadata.append(
                {
                    "dataset": dataset_name,
                    "model": model_label,
                    "split_method": split_method,
                    "cache_dir": str(cache_dir.relative_to(ROOT)),
                    "train_queries": len(train_dataset.queries),
                    "dev_queries": len(dev_dataset.queries),
                    "test_queries": len(test_dataset.queries),
                    "train_missing_positive": train_missing,
                    "dev_missing_positive": dev_missing,
                    "test_missing_positive": test_missing,
                    "pairwise_split_overlap": 0,
                    "rank": args.rank,
                    "selected_alpha": float(linear_model.alpha),
                    "dev_selected_lambda": selected_lambda,
                    "test_gold_used_by_prediction": False,
                    "test_gold_used_for_oracle_diagnostics": True,
                }
            )
            print(
                f"{dataset_name}/{model_label}: alpha={linear_model.alpha:g}, "
                f"dev lambda={selected_lambda:g}, complete",
                flush=True,
            )

    summary = pd.concat(result_frames, ignore_index=True)
    coefficients = pd.DataFrame(coefficient_rows)
    retention = gain_retention_rows(summary)
    selected = pd.concat(selected_rows, ignore_index=True)
    summary.to_csv(args.output_dir / "predicted_metrics_by_lambda.csv", index=False)
    coefficients.to_csv(args.output_dir / "coefficient_prediction_metrics.csv", index=False)
    retention.to_csv(args.output_dir / "gain_retention_by_lambda.csv", index=False)
    selected.to_csv(args.output_dir / "dev_selected_test_metrics.csv", index=False)
    pd.concat(validation_frames, ignore_index=True).to_csv(
        args.output_dir / "ridge_validation.csv", index=False
    )
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    hit10 = summary.pivot(
        index=["dataset", "model", "lambda"],
        columns="method",
        values="HitRate@10",
    ).reset_index()
    hit10.to_csv(args.output_dir / "hit10_curve.csv", index=False)
    for dataset in args.datasets:
        for model_key in args.models:
            model = MODEL_SPECS[model_key]["label"]
            plot_retrieval_curves(
                summary,
                dataset,
                model,
                args.output_dir / f"{dataset}_{model_key}_curves.png",
            )
    plot_coefficient_metrics(
        coefficients, args.output_dir / "coefficient_predictability.png"
    )

    print("\nDev-selected test HitRate@10:")
    print(
        selected.pivot(
            index=["dataset", "model", "dev_selected_lambda"],
            columns="method",
            values="HitRate@10",
        )
        .round(4)
        .to_string()
    )
    print("\nCoefficient predictability:")
    print(coefficients.round(4).to_string(index=False))


if __name__ == "__main__":
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, default=ROOT / "cache")
    parser.add_argument("--results-root", type=Path, default=ROOT / "results")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "coefficient_predictability",
    )
    parser.add_argument("--datasets", nargs="+", default=["fiqa", "arguana"])
    parser.add_argument(
        "--models", nargs="+", choices=MODEL_SPECS, default=list(MODEL_SPECS)
    )
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument(
        "--alphas",
        nargs="+",
        type=float,
        default=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
    )
    parser.add_argument(
        "--lambdas",
        nargs="+",
        type=float,
        default=[0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0],
    )
    parser.add_argument("--batch-size", type=int, default=128)
    run(parser.parse_args())