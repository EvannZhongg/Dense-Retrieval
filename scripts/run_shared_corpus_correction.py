"""Train one correction model per embedding model across all corpora.

Each corpus contributes only document statistics over a shared anchor codebook.  The
correction basis, predictor parameters, and selected correction strength are
shared across corpora for a given embedding model.
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

from run_correction_ablation import (  # noqa: E402
    DEFAULT_LAMBDAS,
    corrected_queries,
    prepare_three_way,
)
from dense_retrieval.analysis.prototype_correction import (  # noqa: E402
    project_correction_targets,
)
from dense_retrieval.analysis.shared_anchors import (  # noqa: E402
    SharedAnchorCodebook,
)
from dense_retrieval.analysis.query_correction import (  # noqa: E402
    build_oracle_deltas,
)
from dense_retrieval.embeddings import MODEL_SPECS  # noqa: E402
from dense_retrieval.evaluation.ranking import METRICS, evaluate_retrieval  # noqa: E402


METHODS = [
    "baseline",
    "global_mean",
    "shared_linear_q_only",
    "shared_linear_sketch",
    "shared_mlp_q_only",
    "shared_mlp_sketch",
]
RIDGE_ALPHAS = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]


def corpus_sketch(queries, codebook, occupancy):
    """Pool fixed-anchor query scores with the current corpus occupancy.

    The final scalar is the weighted log occupancy of the query's Top-M cells;
    it changes when the corpus changes while the anchor projection remains
    fixed for every corpus.
    """
    features = codebook.query_features(queries, occupancy)
    weights = np.exp(features.similarities - features.similarities.max(axis=1, keepdims=True))
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
    selected_anchors = codebook.anchors[features.indices]
    pooled_anchor = np.sum(weights[..., None] * selected_anchors, axis=1)
    pooled_log_occupancy = np.sum(
        weights * occupancy.log_occupancy[features.indices], axis=1, keepdims=True
    )
    return np.concatenate([pooled_anchor, pooled_log_occupancy], axis=1).astype(np.float32)


def model_input(queries, sketches, include_sketch: bool):
    queries = np.asarray(queries, dtype=np.float32)
    second = np.asarray(sketches, dtype=np.float32) if include_sketch else np.zeros_like(queries)
    return np.concatenate([queries, second], axis=1)


def balanced_sample_weights(corpus_names):
    """Give every corpus equal total weight in the pooled training loss."""
    names = np.asarray(corpus_names, dtype=object)
    unique, counts = np.unique(names, return_counts=True)
    count_by_name = dict(zip(unique, counts))
    return np.asarray([len(names) / (len(unique) * count_by_name[name]) for name in names])


def fit_weighted_pca_subspace(deltas, sample_weights, rank: int):
    """Fit a corpus-balanced shared mean and correction basis."""
    weights = np.asarray(sample_weights, dtype=np.float64)
    mean = np.average(deltas, axis=0, weights=weights)
    centered = np.asarray(deltas, dtype=np.float64) - mean
    _, _, components = np.linalg.svd(
        centered * np.sqrt(weights[:, None]), full_matrices=False
    )
    return mean, components[:rank]


def stable_fit_validation_mask(sample_ids, validation_fraction: float = 0.2):
    ordered = sorted(
        sample_ids,
        key=lambda sample_id: hashlib.sha256(
            f"shared-corpus-linear-v1:{sample_id}".encode()
        ).digest(),
    )
    count = max(1, int(round(len(ordered) * validation_fraction)))
    validation_ids = set(ordered[:count])
    validation = np.asarray([sample_id in validation_ids for sample_id in sample_ids])
    return ~validation, validation


def fit_shared_ridge(x, targets, sample_ids, sample_weights):
    fit_mask, validation_mask = stable_fit_validation_mask(sample_ids)
    candidates = []
    for alpha in RIDGE_ALPHAS:
        model = Ridge(alpha=alpha, fit_intercept=True)
        model.fit(x[fit_mask], targets[fit_mask], sample_weight=sample_weights[fit_mask])
        prediction = model.predict(x[validation_mask])
        loss = mean_squared_error(
            targets[validation_mask],
            prediction,
            sample_weight=sample_weights[validation_mask],
        )
        candidates.append((float(loss), float(alpha)))
    alpha = min(candidates)[1]
    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(x, targets, sample_weight=sample_weights)
    return model, alpha


def fit_shared_mlp(x, targets, sample_weights, random_state: int):
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
    model.fit(x, targets, sample_weight=sample_weights)
    return model


def predict_deltas(model, queries, sketches, include_sketch, mean, components):
    coordinates = model.predict(model_input(queries, sketches, include_sketch))
    return mean + coordinates @ components


def evaluate_dev_curves(
    corpora,
    method,
    predictor,
    lambdas,
    batch_size,
):
    rows = []
    for lambda_ in lambdas:
        corpus_values = []
        for dataset_name, data in corpora.items():
            queries = data["dev_queries"]
            deltas = predictor(dataset_name, "dev")
            metrics = evaluate_retrieval(
                data["dev_dataset"],
                corrected_queries(queries, deltas, lambda_),
                data["documents"],
                batch_size,
            )
            corpus_values.append(metrics["HitRate@10"])
            rows.append(
                {
                    "dataset": dataset_name,
                    "method": method,
                    "lambda": float(lambda_),
                    **metrics,
                }
            )
        rows.append(
            {
                "dataset": "macro",
                "method": method,
                "lambda": float(lambda_),
                "HitRate@10": float(np.mean(corpus_values)),
            }
        )
    frame = pd.DataFrame(rows)
    macro = frame[frame["dataset"] == "macro"]
    selected = float(macro.loc[macro["HitRate@10"].idxmax(), "lambda"])
    return selected, frame


def run(args: argparse.Namespace):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    test_rows = []
    dev_frames = []
    metadata = []

    for model_key in args.models:
        corpora = {}
        all_deltas = []
        all_train_queries = []
        all_train_sketches = []
        all_corpus_names = []
        all_sample_ids = []

        loaded = {}
        all_documents = {}
        for dataset_name in args.datasets:
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
            loaded[dataset_name] = {
                "train_dataset": train_dataset,
                "dev_dataset": dev_dataset,
                "test_dataset": test_dataset,
                "train_queries": train_queries,
                "dev_queries": dev_queries,
                "test_queries": test_queries,
                "documents": documents,
                "cache_dir": cache_dir,
                "split_method": split_method,
                "train_rows": train_rows,
                "train_deltas": train_deltas,
                "train_missing": train_missing,
            }
            all_documents[dataset_name] = documents

        codebook = SharedAnchorCodebook.fit(
            all_documents,
            n_anchors=args.n_anchors,
            top_m=args.top_m,
            random_state=args.random_state,
        )
        anchor_occupancies = {
            name: codebook.occupancy(data["documents"])
            for name, data in loaded.items()
        }
        codebook_path = args.output_dir / "models" / f"{model_key}_shared_anchors.npz"
        codebook_path.parent.mkdir(parents=True, exist_ok=True)
        codebook.save(codebook_path, **anchor_occupancies)
        for dataset_name in args.datasets:
            data = loaded[dataset_name]
            train_dataset = data["train_dataset"]
            train_queries = data["train_queries"]
            train_rows = data["train_rows"]
            train_deltas = data["train_deltas"]
            train_missing = data["train_missing"]
            occupancy = anchor_occupancies[dataset_name]
            sketches = {
                "train": corpus_sketch(train_queries[train_rows], codebook, occupancy),
                "dev": corpus_sketch(data["dev_queries"], codebook, occupancy),
                "test": corpus_sketch(data["test_queries"], codebook, occupancy),
            }
            corpora[dataset_name] = {
                "train_dataset": data["train_dataset"],
                "dev_dataset": data["dev_dataset"],
                "test_dataset": data["test_dataset"],
                "train_queries": train_queries[train_rows],
                "dev_queries": data["dev_queries"],
                "test_queries": data["test_queries"],
                "documents": data["documents"],
                "sketches": sketches,
                "cache_dir": data["cache_dir"],
                "split_method": data["split_method"],
                "train_missing": train_missing,
            }
            all_deltas.append(train_deltas)
            all_train_queries.append(train_queries[train_rows])
            all_train_sketches.append(sketches["train"])
            all_corpus_names.extend([dataset_name] * len(train_rows))
            all_sample_ids.extend(
                f"{dataset_name}:{train_dataset.queries[row].query_id}"
                for row in train_rows
            )

        pooled_deltas = np.concatenate(all_deltas)
        sample_weights = balanced_sample_weights(all_corpus_names)
        mean, components = fit_weighted_pca_subspace(
            pooled_deltas, sample_weights, args.rank
        )
        targets = project_correction_targets(pooled_deltas, mean, components)
        train_queries = np.concatenate(all_train_queries).astype(np.float32)
        train_sketches = np.concatenate(all_train_sketches).astype(np.float32)
        q_only_input = model_input(train_queries, train_sketches, False)
        sketch_input = model_input(train_queries, train_sketches, True)

        linear_q, linear_q_alpha = fit_shared_ridge(
            q_only_input, targets, all_sample_ids, sample_weights
        )
        linear_sketch, linear_sketch_alpha = fit_shared_ridge(
            sketch_input, targets, all_sample_ids, sample_weights
        )
        mlp_q = fit_shared_mlp(
            q_only_input, targets, sample_weights, args.random_state
        )
        mlp_sketch = fit_shared_mlp(
            sketch_input, targets, sample_weights, args.random_state
        )

        def make_predictor(method):
            if method == "global_mean":
                return lambda dataset_name, split: np.broadcast_to(
                    mean,
                    (len(corpora[dataset_name][f"{split}_queries"]), len(mean)),
                )
            model, include_sketch = {
                "shared_linear_q_only": (linear_q, False),
                "shared_linear_sketch": (linear_sketch, True),
                "shared_mlp_q_only": (mlp_q, False),
                "shared_mlp_sketch": (mlp_sketch, True),
            }[method]
            return lambda dataset_name, split: predict_deltas(
                model,
                corpora[dataset_name][f"{split}_queries"],
                corpora[dataset_name]["sketches"][split],
                include_sketch,
                mean,
                components,
            )

        selected_lambdas = {"baseline": 0.0}
        for method in METHODS[1:]:
            selected_lambda, frame = evaluate_dev_curves(
                corpora,
                method,
                make_predictor(method),
                args.lambdas,
                args.batch_size,
            )
            frame.insert(0, "model", model_key)
            dev_frames.append(frame)
            selected_lambdas[method] = selected_lambda

        for dataset_name, data in corpora.items():
            baseline = evaluate_retrieval(
                data["test_dataset"],
                data["test_queries"],
                data["documents"],
                args.batch_size,
            )
            test_rows.append(
                {
                    "dataset": dataset_name,
                    "model": model_key,
                    "method": "baseline",
                    "selected_lambda": 0.0,
                    **baseline,
                }
            )
            for method in METHODS[1:]:
                lambda_ = selected_lambdas[method]
                deltas = make_predictor(method)(dataset_name, "test")
                metrics = evaluate_retrieval(
                    data["test_dataset"],
                    corrected_queries(data["test_queries"], deltas, lambda_),
                    data["documents"],
                    args.batch_size,
                )
                test_rows.append(
                    {
                        "dataset": dataset_name,
                        "model": model_key,
                        "method": method,
                        "selected_lambda": lambda_,
                        **metrics,
                    }
                )

        metadata.append(
            {
                "model": model_key,
                "datasets": list(args.datasets),
                "shared_predictors_per_embedding_model": 1,
                "pooled_train_queries": int(sum(len(data["train_queries"]) for data in corpora.values())),
                "corpus_balanced_loss": True,
                "rank": args.rank,
                "n_anchors_global": args.n_anchors,
                "top_m": args.top_m,
                "anchor_codebook_fitted_once_per_embedding_model": True,
                "anchor_codebook": str(codebook_path.relative_to(args.output_dir)),
                "linear_q_alpha": linear_q_alpha,
                "linear_sketch_alpha": linear_sketch_alpha,
                "mlp_hidden_layer_sizes": [256],
                "mlp_q_n_iter": int(mlp_q.n_iter_),
                "mlp_sketch_n_iter": int(mlp_sketch.n_iter_),
                "selected_lambdas_from_macro_dev_HitRate@10": selected_lambdas,
                "test_qrels_used_for_training_or_selection": False,
                "corpora": {
                    name: {
                        "split_method": data["split_method"],
                        "train_queries_used": len(data["train_queries"]),
                        "dev_queries": len(data["dev_queries"]),
                        "test_queries": len(data["test_queries"]),
                        "train_missing_positive": data["train_missing"],
                        "documents": len(data["documents"]),
                    }
                    for name, data in corpora.items()
                },
            }
        )
        print(f"{model_key}: complete", flush=True)

    pd.DataFrame(test_rows).to_csv(
        args.output_dir / "shared_corpus_test_metrics.csv", index=False
    )
    pd.concat(dev_frames, ignore_index=True).to_csv(
        args.output_dir / "shared_corpus_dev_curves.csv", index=False
    )
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=ROOT / "cache")
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "results" / "shared_corpus_correction"
    )
    parser.add_argument(
        "--datasets", nargs="+", default=["fiqa", "arguana", "scifact", "nfcorpus"]
    )
    parser.add_argument(
        "--models", nargs="+", choices=MODEL_SPECS, default=list(MODEL_SPECS)
    )
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--n-anchors", type=int, default=256)
    parser.add_argument("--top-m", type=int, default=16)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--lambdas", nargs="+", type=float, default=DEFAULT_LAMBDAS)
    parser.add_argument("--batch-size", type=int, default=128)
    return parser


if __name__ == "__main__":
    load_dotenv(ROOT / ".env")
    run(build_parser().parse_args())
