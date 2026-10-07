"""Evaluate linear ranking-aligned local-geometry correction under strict LOCO."""
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

from dense_retrieval.analysis.linear_ranking import (  # noqa: E402
    predict_coordinates,
    train_linear_rank_model,
)
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


METHODS = {
    "rank_q_only": "q_only",
    "rank_reference_geometry": "reference_geometry",
    "rank_corpus_geometry": "corpus_geometry",
}


def _fit_balanced_normalizer(values_by_corpus):
    count = len(values_by_corpus)
    center = sum(np.mean(values, axis=0) for values in values_by_corpus.values()) / count
    variance = sum(
        np.mean((values - center) ** 2, axis=0)
        for values in values_by_corpus.values()
    ) / count
    scale = np.sqrt(np.maximum(variance, 1e-12))
    scale = np.where(scale > 1e-6, scale, 1.0)
    return center.astype(np.float32), scale.astype(np.float32)


def _normalize(values, center, scale):
    return np.clip((values - center) / scale, -8.0, 8.0).astype(np.float32)


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


def _features(queries, method, local, reference, components, args):
    return query_correction_features(
        queries,
        method,
        local,
        reference,
        components,
        top_m=args.top_m,
        temperature=args.geometry_temperature,
    )


def run(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_rows = []
    dev_rows = []
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
            reference = fit_reference_geometry(
                {name: loaded[name]["documents"] for name in train_names},
                n_prototypes=args.n_prototypes,
                max_fit_documents=args.max_fit_documents,
                max_iter=args.prototype_max_iter,
                random_state=args.random_state + 1009 * (fold_index + 1),
                assignment_batch_size=args.assignment_batch_size,
            )
            held_data = loaded[heldout]
            baseline = evaluate_retrieval(
                held_data["test_dataset"],
                held_data["test_queries"],
                held_data["documents"],
                args.retrieval_batch_size,
            )
            result_rows.append(
                {
                    "model": model_key,
                    "heldout_corpus": heldout,
                    "method": "baseline",
                    "selected_lambda": 0.0,
                    "coordinate_norm_mean": 0.0,
                    "coordinate_norm_median": 0.0,
                    **baseline,
                }
            )

            fold_metadata = {
                "model": model_key,
                "heldout_corpus": heldout,
                "training_corpora": train_names,
                "rank": len(components),
                "n_prototypes": args.n_prototypes,
                "top_m": args.top_m,
                "geometry_temperature": args.geometry_temperature,
                "hard_negatives": args.hard_negatives,
                "ranking_temperature": args.ranking_temperature,
                "epochs": args.epochs,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "coordinate_penalty": args.coordinate_penalty,
                "train_lambdas": list(args.train_lambdas),
                "lambda_grid": list(args.lambdas),
                "negative_refresh": "static_frozen_baseline_topk",
                "prototype_fit_is_document_only": True,
                "heldout_qrels_used_for_training_or_selection": False,
                "methods": {},
            }
            for result_method, feature_method in METHODS.items():
                raw_train = {
                    name: _features(
                        loaded[name]["train_queries"],
                        feature_method,
                        local_geometries[name],
                        reference,
                        components,
                        args,
                    )
                    for name in train_names
                }
                center, scale = _fit_balanced_normalizer(raw_train)
                corpora = {}
                for name in train_names:
                    data = loaded[name]
                    corpora[name] = {
                        "train_dataset": data["train_dataset"],
                        "train_queries": data["train_queries"],
                        "train_features": _normalize(raw_train[name], center, scale),
                        "documents": data["documents"],
                    }
                model, losses = train_linear_rank_model(
                    corpora,
                    components,
                    epochs=args.epochs,
                    batch_size=args.train_batch_size,
                    train_lambdas=args.train_lambdas,
                    temperature=args.ranking_temperature,
                    learning_rate=args.learning_rate,
                    weight_decay=args.weight_decay,
                    coordinate_penalty=args.coordinate_penalty,
                    hard_negatives=args.hard_negatives,
                    random_state=args.random_state,
                    device=args.device,
                )

                curve = []
                for lambda_ in args.lambdas:
                    values = []
                    for name in train_names:
                        data = loaded[name]
                        raw = _features(
                            data["dev_queries"],
                            feature_method,
                            local_geometries[name],
                            reference,
                            components,
                            args,
                        )
                        coordinates = predict_coordinates(
                            model, _normalize(raw, center, scale)
                        )
                        metrics = evaluate_retrieval(
                            data["dev_dataset"],
                            corrected_queries(
                                data["dev_queries"],
                                coordinates @ components,
                                lambda_,
                            ),
                            data["documents"],
                            args.retrieval_batch_size,
                        )
                        values.append(metrics["HitRate@10"])
                    macro = float(np.mean(values))
                    curve.append((float(lambda_), macro))
                    dev_rows.append(
                        {
                            "model": model_key,
                            "heldout_corpus": heldout,
                            "method": result_method,
                            "lambda": float(lambda_),
                            "macro_dev_HitRate@10": macro,
                        }
                    )
                selected_lambda = max(
                    curve, key=lambda value: (value[1], -value[0])
                )[0]
                held_raw = _features(
                    held_data["test_queries"],
                    feature_method,
                    local_geometries[heldout],
                    reference,
                    components,
                    args,
                )
                held_coordinates = predict_coordinates(
                    model, _normalize(held_raw, center, scale)
                )
                metrics = evaluate_retrieval(
                    held_data["test_dataset"],
                    corrected_queries(
                        held_data["test_queries"],
                        held_coordinates @ components,
                        selected_lambda,
                    ),
                    held_data["documents"],
                    args.retrieval_batch_size,
                )
                norms = np.linalg.norm(held_coordinates, axis=1)
                result_rows.append(
                    {
                        "model": model_key,
                        "heldout_corpus": heldout,
                        "method": result_method,
                        "selected_lambda": selected_lambda,
                        "coordinate_norm_mean": float(np.mean(norms)),
                        "coordinate_norm_median": float(np.median(norms)),
                        **metrics,
                    }
                )
                fold_metadata["methods"][result_method] = {
                    "feature_dimension": int(raw_train[train_names[0]].shape[1]),
                    "parameter_count": int(
                        sum(parameter.numel() for parameter in model.parameters())
                    ),
                    "loss_history": losses,
                    "selected_lambda": selected_lambda,
                    "device": str(next(model.parameters()).device),
                }
            metadata.append(fold_metadata)
            print(f"{model_key}/{heldout}: complete", flush=True)

    pd.DataFrame(result_rows).to_csv(
        args.output_dir / "rank_local_geometry_test_metrics.csv", index=False
    )
    pd.DataFrame(dev_rows).to_csv(
        args.output_dir / "rank_local_geometry_dev_curves.csv", index=False
    )
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=ROOT / "cache")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "rank_local_geometry_loco",
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
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--n-prototypes", type=int, default=16)
    parser.add_argument("--top-m", type=int, default=8)
    parser.add_argument("--geometry-temperature", type=float, default=0.07)
    parser.add_argument("--max-fit-documents", type=int, default=5000)
    parser.add_argument("--prototype-max-iter", type=int, default=10)
    parser.add_argument("--assignment-batch-size", type=int, default=4096)
    parser.add_argument("--hard-negatives", type=int, default=32)
    parser.add_argument("--ranking-temperature", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--train-batch-size", type=int, default=128)
    parser.add_argument("--retrieval-batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--coordinate-penalty", type=float, default=0.1)
    parser.add_argument(
        "--train-lambdas",
        nargs="+",
        type=float,
        default=[0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0],
    )
    parser.add_argument("--lambdas", nargs="+", type=float, default=DEFAULT_LAMBDAS)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser


if __name__ == "__main__":
    load_dotenv(ROOT / ".env")
    run(build_parser().parse_args())
