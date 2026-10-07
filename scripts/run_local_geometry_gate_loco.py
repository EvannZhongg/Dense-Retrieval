"""Test whether corpus-local geometry can gate one fixed query-only correction."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.linear_model import Ridge

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
from dense_retrieval.evaluation.quality import compute_quality_metrics  # noqa: E402
from dense_retrieval.evaluation.ranking import (  # noqa: E402
    evaluate_retrieval,
    top_k_document_ids,
)


GATE_METHODS = {
    "gate_q_only": "q_only",
    "gate_reference_geometry": "reference_geometry",
    "gate_corpus_geometry": "corpus_geometry",
}


def _balanced_weights(sizes):
    return np.concatenate(
        [np.full(size, 1.0 / (len(sizes) * size)) for size in sizes]
    )


def _fit_normalizer(values_by_corpus):
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


def _per_query_ndcg(dataset, queries, documents, batch_size):
    rankings = top_k_document_ids(
        queries, documents, list(dataset.corpus), 10, batch_size
    )
    _metrics, frame = compute_quality_metrics(
        [sample.query_id for sample in dataset.queries],
        dataset.qrels,
        rankings,
        cutoffs=(5, 10),
        available_document_ids=set(dataset.corpus),
    )
    return frame["ndcg_at_10"].to_numpy(dtype=np.float32)


def _gate_features(queries, method, local, reference, components, args):
    return query_correction_features(
        queries,
        method,
        local,
        reference,
        components,
        top_m=args.top_m,
        temperature=args.geometry_temperature,
    )


def _gated_queries(original, corrected, mask):
    result = np.asarray(original, dtype=np.float32).copy()
    result[np.asarray(mask, dtype=bool)] = np.asarray(corrected, dtype=np.float32)[
        np.asarray(mask, dtype=bool)
    ]
    return result


def _thresholds(prediction):
    finite = np.asarray(prediction, dtype=np.float64)
    quantiles = np.quantile(finite, [0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0])
    return np.unique(np.concatenate(([-np.inf, 0.0], quantiles, [np.inf])))


def run(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_rows = []
    selection_rows = []
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

            candidate_train_x = []
            candidate_train_y = []
            candidate_sizes = []
            for name in train_names:
                data = loaded[name]
                rows = data["train_rows"]
                candidate_train_x.append(data["train_queries"][rows])
                candidate_train_y.append(
                    (data["train_deltas"] @ components.T).astype(np.float32)
                )
                candidate_sizes.append(len(rows))
            candidate_x = np.concatenate(candidate_train_x)
            candidate_y = np.concatenate(candidate_train_y)
            candidate_weights = _balanced_weights(candidate_sizes)
            candidate_center, candidate_scale = _fit_normalizer(
                {str(index): values for index, values in enumerate(candidate_train_x)}
            )
            candidate_model = Ridge(alpha=args.candidate_alpha)
            candidate_model.fit(
                _normalize(candidate_x, candidate_center, candidate_scale),
                candidate_y,
                sample_weight=candidate_weights * len(candidate_weights),
            )

            split_state = {}
            required_splits = {
                **{name: ("train", "dev") for name in train_names},
                heldout: ("test",),
            }
            for name, splits in required_splits.items():
                data = loaded[name]
                split_state[name] = {}
                for split in splits:
                    queries = data[f"{split}_queries"]
                    dataset = data[f"{split}_dataset"]
                    coordinates = candidate_model.predict(
                        _normalize(queries, candidate_center, candidate_scale)
                    )
                    split_state[name][split] = {
                        "queries": queries,
                        "dataset": dataset,
                        "coordinates": coordinates.astype(np.float32),
                        "baseline_ndcg": _per_query_ndcg(
                            dataset, queries, data["documents"], args.batch_size
                        ),
                    }

            lambda_curve = []
            for lambda_ in args.lambdas:
                corpus_values = []
                for name in train_names:
                    state = split_state[name]["dev"]
                    corrected = corrected_queries(
                        state["queries"], state["coordinates"] @ components, lambda_
                    )
                    corrected_ndcg = _per_query_ndcg(
                        state["dataset"],
                        corrected,
                        loaded[name]["documents"],
                        args.batch_size,
                    )
                    corpus_values.append(
                        float(np.mean(np.maximum(state["baseline_ndcg"], corrected_ndcg)))
                    )
                macro = float(np.mean(corpus_values))
                lambda_curve.append((float(lambda_), macro))
                selection_rows.append(
                    {
                        "model": model_key,
                        "heldout_corpus": heldout,
                        "selection": "oracle_candidate_lambda",
                        "method": "query_only_candidate",
                        "hyperparameter": float(lambda_),
                        "threshold": np.nan,
                        "macro_dev_NDCG@10": macro,
                    }
                )
            selected_lambda = max(
                lambda_curve, key=lambda value: (value[1], -value[0])
            )[0]

            for name, splits in required_splits.items():
                data = loaded[name]
                for split in splits:
                    state = split_state[name][split]
                    state["corrected_queries"] = corrected_queries(
                        state["queries"],
                        state["coordinates"] @ components,
                        selected_lambda,
                    )
                    state["corrected_ndcg"] = _per_query_ndcg(
                        state["dataset"],
                        state["corrected_queries"],
                        data["documents"],
                        args.batch_size,
                    )
                    state["utility_gain"] = (
                        state["corrected_ndcg"] - state["baseline_ndcg"]
                    )

            held_data = loaded[heldout]
            held_state = split_state[heldout]["test"]
            baseline_metrics = evaluate_retrieval(
                held_state["dataset"],
                held_state["queries"],
                held_data["documents"],
                args.batch_size,
            )
            always_metrics = evaluate_retrieval(
                held_state["dataset"],
                held_state["corrected_queries"],
                held_data["documents"],
                args.batch_size,
            )
            for method, metrics, apply_rate in (
                ("baseline", baseline_metrics, 0.0),
                ("always_query_correction", always_metrics, 1.0),
            ):
                result_rows.append(
                    {
                        "model": model_key,
                        "heldout_corpus": heldout,
                        "method": method,
                        "candidate_lambda": selected_lambda,
                        "gate_alpha": np.nan,
                        "gate_threshold": np.nan,
                        "apply_rate": apply_rate,
                        "applied_benefit_rate": np.nan,
                        "applied_harm_rate": np.nan,
                        **metrics,
                    }
                )

            oracle_mask = held_state["utility_gain"] > 0
            oracle_metrics = evaluate_retrieval(
                held_state["dataset"],
                _gated_queries(
                    held_state["queries"],
                    held_state["corrected_queries"],
                    oracle_mask,
                ),
                held_data["documents"],
                args.batch_size,
            )
            result_rows.append(
                {
                    "model": model_key,
                    "heldout_corpus": heldout,
                    "method": "oracle_gate_diagnostic",
                    "candidate_lambda": selected_lambda,
                    "gate_alpha": np.nan,
                    "gate_threshold": np.nan,
                    "apply_rate": float(np.mean(oracle_mask)),
                    "applied_benefit_rate": 1.0 if np.any(oracle_mask) else np.nan,
                    "applied_harm_rate": 0.0 if np.any(oracle_mask) else np.nan,
                    **oracle_metrics,
                }
            )

            fold_methods = {}
            for result_method, feature_method in GATE_METHODS.items():
                raw_train = {
                    name: _gate_features(
                        loaded[name]["train_queries"],
                        feature_method,
                        local_geometries[name],
                        reference,
                        components,
                        args,
                    )
                    for name in train_names
                }
                center, scale = _fit_normalizer(raw_train)
                train_x = np.concatenate(
                    [_normalize(raw_train[name], center, scale) for name in train_names]
                )
                train_y = np.concatenate(
                    [split_state[name]["train"]["utility_gain"] for name in train_names]
                )
                sizes = [len(raw_train[name]) for name in train_names]
                weights = _balanced_weights(sizes)

                best = None
                best_model = None
                best_threshold = None
                for alpha in args.gate_alphas:
                    model = Ridge(alpha=alpha)
                    model.fit(
                        train_x,
                        train_y,
                        sample_weight=weights * len(weights),
                    )
                    train_prediction = model.predict(train_x)
                    for threshold in _thresholds(train_prediction):
                        corpus_scores = []
                        for name in train_names:
                            data = loaded[name]
                            raw_dev = _gate_features(
                                data["dev_queries"],
                                feature_method,
                                local_geometries[name],
                                reference,
                                components,
                                args,
                            )
                            prediction = model.predict(
                                _normalize(raw_dev, center, scale)
                            )
                            state = split_state[name]["dev"]
                            chosen = np.where(
                                prediction > threshold,
                                state["corrected_ndcg"],
                                state["baseline_ndcg"],
                            )
                            corpus_scores.append(float(np.mean(chosen)))
                        macro = float(np.mean(corpus_scores))
                        selection_rows.append(
                            {
                                "model": model_key,
                                "heldout_corpus": heldout,
                                "selection": "learned_gate",
                                "method": result_method,
                                "hyperparameter": float(alpha),
                                "threshold": float(threshold),
                                "macro_dev_NDCG@10": macro,
                            }
                        )
                        candidate = (macro, float(threshold))
                        if best is None or candidate > best:
                            best = candidate
                            best_model = model
                            best_threshold = float(threshold)

                held_raw = _gate_features(
                    held_data["test_queries"],
                    feature_method,
                    local_geometries[heldout],
                    reference,
                    components,
                    args,
                )
                held_prediction = best_model.predict(
                    _normalize(held_raw, center, scale)
                )
                mask = held_prediction > best_threshold
                metrics = evaluate_retrieval(
                    held_state["dataset"],
                    _gated_queries(
                        held_state["queries"],
                        held_state["corrected_queries"],
                        mask,
                    ),
                    held_data["documents"],
                    args.batch_size,
                )
                applied_gain = held_state["utility_gain"][mask]
                result_rows.append(
                    {
                        "model": model_key,
                        "heldout_corpus": heldout,
                        "method": result_method,
                        "candidate_lambda": selected_lambda,
                        "gate_alpha": float(best_model.alpha),
                        "gate_threshold": best_threshold,
                        "apply_rate": float(np.mean(mask)),
                        "applied_benefit_rate": (
                            float(np.mean(applied_gain > 0)) if np.any(mask) else np.nan
                        ),
                        "applied_harm_rate": (
                            float(np.mean(applied_gain < 0)) if np.any(mask) else np.nan
                        ),
                        **metrics,
                    }
                )
                fold_methods[result_method] = {
                    "feature_dimension": int(train_x.shape[1]),
                    "selected_alpha": float(best_model.alpha),
                    "selected_threshold": best_threshold,
                    "macro_dev_NDCG@10": float(best[0]),
                }

            metadata.append(
                {
                    "model": model_key,
                    "heldout_corpus": heldout,
                    "training_corpora": train_names,
                    "rank": len(components),
                    "candidate": "query_only_nearest_positive_ridge",
                    "candidate_alpha": args.candidate_alpha,
                    "candidate_lambda_selection": "training_corpora_dev_oracle_gate_NDCG@10",
                    "selected_candidate_lambda": selected_lambda,
                    "gate_target": "per_query_corrected_minus_baseline_NDCG@10",
                    "gate_model": "ridge_utility_regression",
                    "gate_alphas": list(args.gate_alphas),
                    "prototype_fit_is_document_only": True,
                    "heldout_qrels_used_for_training_or_selection": False,
                    "heldout_oracle_gate_is_diagnostic_only": True,
                    "methods": fold_methods,
                }
            )
            print(f"{model_key}/{heldout}: complete", flush=True)

    pd.DataFrame(result_rows).to_csv(
        args.output_dir / "local_geometry_gate_test_metrics.csv", index=False
    )
    pd.DataFrame(selection_rows).to_csv(
        args.output_dir / "local_geometry_gate_selection.csv", index=False
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
        default=ROOT / "results" / "local_geometry_gate_loco",
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
    parser.add_argument("--candidate-alpha", type=float, default=10_000.0)
    parser.add_argument(
        "--gate-alphas", nargs="+", type=float, default=[100.0, 1000.0, 10_000.0]
    )
    parser.add_argument("--n-prototypes", type=int, default=16)
    parser.add_argument("--top-m", type=int, default=8)
    parser.add_argument("--geometry-temperature", type=float, default=0.07)
    parser.add_argument("--max-fit-documents", type=int, default=5000)
    parser.add_argument("--prototype-max-iter", type=int, default=10)
    parser.add_argument("--assignment-batch-size", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lambdas", nargs="+", type=float, default=DEFAULT_LAMBDAS)
    parser.add_argument("--random-state", type=int, default=0)
    return parser


if __name__ == "__main__":
    load_dotenv(ROOT / ".env")
    run(build_parser().parse_args())
