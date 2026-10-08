"""Predict privileged ranking coordinates under leave-one-corpus-out transfer.

This experiment separates action-space capacity from target learnability.  A
shared ranking-boundary basis is fitted from training corpora only.  Training
queries receive per-query coordinates from a privileged ranking oracle; Ridge
models then predict those coordinates from either the query alone or query
interactions with a document-only spectral summary.  Held-out qrels are used
only for final retrieval metrics.
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dense_retrieval.analysis.action_space import (  # noqa: E402
    fit_svd_subspace,
    optimize_ranking_oracle,
    ranking_boundary_directions,
)
from dense_retrieval.analysis.corpus_spectrum import (  # noqa: E402
    document_spectrum,
    fit_shared_document_basis,
    query_spectral_interactions,
)
from dense_retrieval.analysis.ranking_correction import build_rank_records  # noqa: E402
from dense_retrieval.analysis.study_data import prepare_three_way  # noqa: E402
from dense_retrieval.embeddings import MODEL_SPECS  # noqa: E402
from dense_retrieval.evaluation.ranking import evaluate_retrieval  # noqa: E402


METHODS = ("q_only", "corpus_spectral")


def _prepare(dataset_name: str, model_key: str, cache_root: Path):
    (
        train_dataset,
        dev_dataset,
        test_dataset,
        train_queries,
        dev_queries,
        test_queries,
        documents,
        cache_dir,
        split,
    ) = prepare_three_way(
        dataset_name,
        model_key,
        cache_root,
        datasets_root=ROOT / "datasets",
        configs_root=ROOT / "configs",
    )
    return {
        "train_dataset": train_dataset,
        "dev_dataset": dev_dataset,
        "test_dataset": test_dataset,
        "train_queries": train_queries,
        "dev_queries": dev_queries,
        "test_queries": test_queries,
        "documents": documents,
        "metadata": {
            "split": split,
            "cache_dir": str(cache_dir.relative_to(ROOT)),
            "train_queries": len(train_dataset.queries),
            "dev_queries": len(dev_dataset.queries),
            "test_queries": len(test_dataset.queries),
        },
    }


def _oracle_coordinates(data, split: str, components: np.ndarray, args):
    dataset = data[f"{split}_dataset"]
    queries = data[f"{split}_queries"]
    records = build_rank_records(dataset, queries, data["documents"], args.hard_negatives)
    rows, deltas, rank_loss = optimize_ranking_oracle(
        queries,
        data["documents"],
        records,
        components,
        steps=args.oracle_steps,
        learning_rate=args.oracle_learning_rate,
        penalty=args.oracle_penalty,
        temperature=args.temperature,
        device=args.device,
    )
    coordinates = deltas @ np.asarray(components, dtype=np.float64).T
    return rows, coordinates, rank_loss


def _features(data, split: str, method: str, summary, global_mean, global_components):
    queries = np.asarray(data[f"{split}_queries"], dtype=np.float32)
    if method == "q_only":
        return queries
    interactions = query_spectral_interactions(
        queries, summary, global_mean, global_components
    )
    return np.concatenate([queries, interactions], axis=1).astype(np.float32)


def _fit_model(train_x, train_y, dev_x, dev_y, alphas):
    validation = []
    for alpha in alphas:
        model = Pipeline(
            [
                ("scale", StandardScaler()),
                ("ridge", Ridge(alpha=float(alpha), solver="lsqr")),
            ]
        )
        model.fit(train_x, train_y)
        prediction = model.predict(dev_x)
        validation.append(
            {
                "alpha": float(alpha),
                "dev_coordinate_mse": float(np.mean((prediction - dev_y) ** 2)),
            }
        )
    selected = min(validation, key=lambda row: (row["dev_coordinate_mse"], row["alpha"]))
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            ("ridge", Ridge(alpha=selected["alpha"], solver="lsqr")),
        ]
    )
    model.fit(train_x, train_y)
    return model, pd.DataFrame(validation), selected["alpha"]


def _select_lambda(corpora, predictions, lambdas, batch_size):
    rows = []
    for lambda_ in lambdas:
        values = []
        for name, data in corpora.items():
            corrected = _apply(data["dev_queries"], predictions[name]["dev"], lambda_)
            metrics = evaluate_retrieval(
                data["dev_dataset"], corrected, data["documents"], batch_size
            )
            values.append(metrics["NDCG@10"])
            rows.append({"corpus": name, "lambda": lambda_, **metrics})
        rows.append({"corpus": "macro", "lambda": lambda_, "NDCG@10": float(np.mean(values))})
    frame = pd.DataFrame(rows)
    macro = frame[frame["corpus"] == "macro"]
    selected = float(macro.loc[macro["NDCG@10"].idxmax(), "lambda"])
    return selected, frame


def _apply(queries, deltas, lambda_):
    corrected = np.asarray(queries, dtype=np.float64) + float(lambda_) * np.asarray(
        deltas, dtype=np.float64
    )
    corrected /= np.maximum(np.linalg.norm(corrected, axis=1, keepdims=True), 1e-12)
    return corrected.astype(np.float32)


def run(args):
    if len(args.datasets) < 2:
        raise ValueError("at least two corpora are required")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    test_rows = []
    validation_rows = []
    metadata = []

    for model_key in args.models:
        model_label = MODEL_SPECS[model_key]["label"]
        all_data = {name: _prepare(name, model_key, args.cache_root) for name in args.datasets}
        for heldout in args.datasets:
            training = [name for name in args.datasets if name != heldout]
            train_boundary = []
            for name in training:
                _, boundary = ranking_boundary_directions(
                    all_data[name]["train_dataset"],
                    all_data[name]["train_queries"],
                    all_data[name]["documents"],
                    hard_negatives=args.hard_negatives,
                    temperature=args.temperature,
                )
                train_boundary.append(boundary)
            components = fit_svd_subspace(np.concatenate(train_boundary), args.rank)
            global_mean, global_components = fit_shared_document_basis(
                [all_data[name]["documents"] for name in training],
                args.spectrum_rank,
                max_documents_per_set=args.spectrum_samples,
                random_state=args.random_state,
            )
            summaries = {
                name: document_spectrum(
                    all_data[name]["documents"], global_mean, global_components
                )
                for name in args.datasets
            }

            targets = {}
            oracle_losses = {}
            for name in training:
                train_rows, train_coords, train_loss = _oracle_coordinates(
                    all_data[name], "train", components, args
                )
                dev_rows, dev_coords, dev_loss = _oracle_coordinates(
                    all_data[name], "dev", components, args
                )
                targets[name] = {
                    "train_rows": train_rows,
                    "train": train_coords,
                    "dev_rows": dev_rows,
                    "dev": dev_coords,
                }
                oracle_losses[name] = {"train": train_loss, "dev": dev_loss}

            models = {}
            predictions = {method: {} for method in METHODS}
            for method in METHODS:
                train_x = np.concatenate(
                    [
                        _features(all_data[name], "train", method, summaries[name], global_mean, global_components)[targets[name]["train_rows"]]
                        for name in training
                    ],
                    axis=0,
                )
                train_y = np.concatenate([targets[name]["train"] for name in training], axis=0)
                dev_x = np.concatenate(
                    [
                        _features(all_data[name], "dev", method, summaries[name], global_mean, global_components)[targets[name]["dev_rows"]]
                        for name in training
                    ],
                    axis=0,
                )
                dev_y = np.concatenate([targets[name]["dev"] for name in training], axis=0)
                model, validation, alpha = _fit_model(
                    train_x, train_y, dev_x, dev_y, args.alphas
                )
                validation.insert(0, "method", method)
                validation.insert(0, "heldout_corpus", heldout)
                validation_rows.extend(validation.to_dict("records"))
                models[method] = {"model": model, "alpha": alpha}
                for name in args.datasets:
                    values = _features(
                        all_data[name], "dev", method, summaries[name], global_mean, global_components
                    )
                    predictions[method][name] = {
                        "dev": model.predict(values) @ components,
                    }

            selected_lambdas = {}
            lambda_tables = {}
            for method in METHODS:
                selected_lambdas[method], lambda_tables[method] = _select_lambda(
                    {name: all_data[name] for name in training},
                    {name: predictions[method][name] for name in training},
                    args.lambdas,
                    args.batch_size,
                )

            for method in METHODS:
                model = models[method]["model"]
                test_values = _features(
                    all_data[heldout], "test", method, summaries[heldout], global_mean, global_components
                )
                predicted_coordinates = model.predict(test_values)
                predicted_deltas = predicted_coordinates @ components
                lambda_ = selected_lambdas[method]
                corrected = _apply(all_data[heldout]["test_queries"], predicted_deltas, lambda_)
                metrics = evaluate_retrieval(
                    all_data[heldout]["test_dataset"],
                    corrected,
                    all_data[heldout]["documents"],
                    args.batch_size,
                )
                test_rows.append(
                    {
                        "heldout_corpus": heldout,
                        "model": model_label,
                        "method": method,
                        "rank": args.rank,
                        "spectrum_rank": args.spectrum_rank,
                        "selected_alpha": models[method]["alpha"],
                        "selected_lambda": lambda_,
                        **metrics,
                    }
                )
            baseline = evaluate_retrieval(
                all_data[heldout]["test_dataset"],
                all_data[heldout]["test_queries"],
                all_data[heldout]["documents"],
                args.batch_size,
            )
            test_rows.append(
                {
                    "heldout_corpus": heldout,
                    "model": model_label,
                    "method": "baseline",
                    "rank": 0,
                    "spectrum_rank": args.spectrum_rank,
                    "selected_alpha": np.nan,
                    "selected_lambda": 0.0,
                    **baseline,
                }
            )
            metadata.append(
                {
                    "model": model_label,
                    "heldout_corpus": heldout,
                    "training_corpora": training,
                    "rank": args.rank,
                    "spectrum_rank": args.spectrum_rank,
                    "basis": "ranking_boundary_svd fitted on training qrels only",
                    "corpus_features": "document-only shared spectral basis plus corpus mean/variance interactions",
                    "heldout_qrels_used_for_features": False,
                    "heldout_qrels_used_for_training_or_selection": False,
                    "oracle_losses_training_corpora": oracle_losses,
                    "selected_lambdas": selected_lambdas,
                }
            )
            print(f"{model_key}/{heldout}: complete", flush=True)

    result = pd.DataFrame(test_rows)
    result.to_csv(args.output_dir / "direct_coordinate_test_metrics.csv", index=False)
    pd.DataFrame(validation_rows).to_csv(
        args.output_dir / "direct_coordinate_validation.csv", index=False
    )
    (args.output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "protocol": "leave-one-corpus-out direct ranking-coordinate prediction",
                "methods": list(METHODS),
                "folds": metadata,
                "alphas": args.alphas,
                "lambdas": args.lambdas,
                "hard_negatives": args.hard_negatives,
                "temperature": args.temperature,
                "oracle_steps": args.oracle_steps,
                "device": args.device,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(result.round(4).to_string(index=False), flush=True)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=ROOT / "cache")
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "results" / "direct_coordinate_loco"
    )
    parser.add_argument("--datasets", nargs="+", default=["fiqa", "arguana", "scifact", "nfcorpus"])
    parser.add_argument(
        "--models", nargs="+", choices=MODEL_SPECS, default=["text-embedding-3-small-aiberm"]
    )
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--spectrum-rank", type=int, default=32)
    parser.add_argument("--spectrum-samples", type=int, default=5000)
    parser.add_argument("--hard-negatives", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--oracle-steps", type=int, default=100)
    parser.add_argument("--oracle-learning-rate", type=float, default=0.05)
    parser.add_argument("--oracle-penalty", type=float, default=1e-3)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--alphas", nargs="+", type=float, default=[1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0])
    parser.add_argument("--lambdas", nargs="+", type=float, default=[0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--random-state", type=int, default=0)
    return parser


if __name__ == "__main__":
    load_dotenv(ROOT / ".env")
    run(build_parser().parse_args())
