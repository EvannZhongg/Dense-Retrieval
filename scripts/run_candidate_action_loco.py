"""Score discrete correction actions by retrieval utility under strict LOCO.

Each fold learns a ranking-boundary basis and a small corpus-balanced action
codebook from labeled training corpora.  Ridge utility scorers then compare
query-only, fixed-reference, and corpus-local geometry features.  The held-out
corpus contributes document embeddings only until the final policy and oracle
candidate diagnostics are evaluated.
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
    optimize_ranking_oracle_solution,
    ranking_boundary_directions,
)
from dense_retrieval.analysis.candidate_actions import (  # noqa: E402
    action_query_features,
    corrected_query_grid,
    fit_candidate_actions,
    reshape_utilities,
)
from dense_retrieval.analysis.local_geometry import (  # noqa: E402
    CorpusPrototypeGeometry,
    fit_reference_geometry,
    query_local_geometry_features,
)
from dense_retrieval.analysis.ranking_correction import build_rank_records  # noqa: E402
from dense_retrieval.analysis.study_data import prepare_three_way  # noqa: E402
from dense_retrieval.embeddings import MODEL_SPECS  # noqa: E402
from dense_retrieval.evaluation.quality import compute_quality_metrics  # noqa: E402
from dense_retrieval.evaluation.ranking import evaluate_retrieval  # noqa: E402


METHODS = ("q_only", "reference_geometry", "corpus_geometry")


def _prepare(dataset_name: str, model_key: str, cache_root: Path):
    values = prepare_three_way(
        dataset_name,
        model_key,
        cache_root,
        datasets_root=ROOT / "datasets",
        configs_root=ROOT / "configs",
    )
    keys = (
        "train_dataset", "dev_dataset", "test_dataset", "train_queries",
        "dev_queries", "test_queries", "documents", "cache_dir", "split",
    )
    return dict(zip(keys, values))


def _load_or_fit_geometry(output_dir, model_key, corpus_name, documents, args):
    path = output_dir / "models" / (
        f"{model_key}_{corpus_name}_k{args.n_prototypes}_seed{args.random_state}.npz"
    )
    if path.exists():
        with np.load(path) as values:
            return CorpusPrototypeGeometry(values["prototypes"], values["occupancy"])
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
    )
    return geometry


def _top_k_indices(scores: np.ndarray, k: int) -> np.ndarray:
    width = min(int(k), scores.shape[1])
    candidates = np.argpartition(-scores, width - 1, axis=1)[:, :width]
    values = np.take_along_axis(scores, candidates, axis=1)
    order = np.argsort(-values, axis=1, kind="mergesort")
    return np.take_along_axis(candidates, order, axis=1)


def _candidate_quality(dataset, queries, documents, actions, components, batch_size):
    """Evaluate all actions while computing the expensive query-doc dot once."""
    query_values = np.asarray(queries, dtype=np.float32)
    document_values = np.asarray(documents, dtype=np.float32)
    deltas = np.asarray(actions, dtype=np.float32) @ np.asarray(components, dtype=np.float32)
    shifts = deltas @ document_values.T
    rankings = [[] for _ in range(len(actions))]
    for start in range(0, len(query_values), int(batch_size)):
        baseline_scores = query_values[start : start + batch_size] @ document_values.T
        for action_index in range(len(actions)):
            rankings[action_index].extend(
                _top_k_indices(baseline_scores + shifts[action_index], 10).tolist()
            )
    document_ids = np.asarray(list(dataset.corpus), dtype=object)
    query_ids = [sample.query_id for sample in dataset.queries]
    ndcg = np.empty((len(query_values), len(actions)), dtype=np.float32)
    for action_index, indices in enumerate(rankings):
        _metrics, frame = compute_quality_metrics(
            query_ids,
            dataset.qrels,
            document_ids[np.asarray(indices)].tolist(),
            cutoffs=(5, 10),
            available_document_ids=set(dataset.corpus),
        )
        ndcg[:, action_index] = frame["ndcg_at_10"].to_numpy(dtype=np.float32)
    return ndcg


def _action_features(queries, actions, components, method, geometry, args):
    base = action_query_features(queries, actions, components)
    if method == "q_only":
        return base
    corrected = corrected_query_grid(queries, actions, components)
    original = query_local_geometry_features(
        queries,
        geometry,
        components,
        top_m=min(args.top_m, len(geometry.prototypes)),
        temperature=args.geometry_temperature,
    )
    response = query_local_geometry_features(
        corrected,
        geometry,
        components,
        top_m=min(args.top_m, len(geometry.prototypes)),
        temperature=args.geometry_temperature,
    )
    repeated = np.repeat(original, len(actions), axis=0)
    return np.concatenate([base, repeated, response - repeated], axis=1).astype(np.float32)


def _fit_utility_model(train_features, train_targets, corpus_sizes, alpha):
    weights = np.concatenate(
        [np.full(size, 1.0 / (len(corpus_sizes) * size)) for size in corpus_sizes]
    )
    model = Pipeline(
        [("scale", StandardScaler()), ("ridge", Ridge(alpha=float(alpha), solver="lsqr"))]
    )
    model.fit(train_features, train_targets, ridge__sample_weight=weights * len(weights))
    return model


def _choose_actions(predicted, action_count, threshold):
    scores = reshape_utilities(predicted, len(predicted) // action_count, action_count)
    best = np.argmax(scores[:, 1:], axis=1) + 1
    best_scores = scores[np.arange(len(scores)), best]
    return np.where(best_scores > float(threshold), best, 0)


def _selected_queries(queries, choices, actions, components):
    deltas = np.asarray(actions) @ np.asarray(components)
    corrected = np.asarray(queries, dtype=np.float64) + deltas[np.asarray(choices)]
    corrected /= np.maximum(np.linalg.norm(corrected, axis=1, keepdims=True), 1e-12)
    return corrected.astype(np.float32)


def run(args):
    if len(args.datasets) < 2:
        raise ValueError("at least two corpora are required")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_rows = []
    selection_rows = []
    fold_metadata = []

    for model_key in args.models:
        loaded = {name: _prepare(name, model_key, args.cache_root) for name in args.datasets}
        local_geometries = {
            name: _load_or_fit_geometry(
                args.output_dir, model_key, name, loaded[name]["documents"], args
            )
            for name in args.datasets
        }
        for heldout in args.datasets:
            training = [name for name in args.datasets if name != heldout]
            boundaries = []
            for name in training:
                _, directions = ranking_boundary_directions(
                    loaded[name]["train_dataset"],
                    loaded[name]["train_queries"],
                    loaded[name]["documents"],
                    hard_negatives=args.hard_negatives,
                    temperature=args.ranking_temperature,
                )
                boundaries.append(directions)
            components = fit_svd_subspace(np.concatenate(boundaries), args.rank)

            oracle_coordinate_sets = []
            for name in training:
                records = build_rank_records(
                    loaded[name]["train_dataset"],
                    loaded[name]["train_queries"],
                    loaded[name]["documents"],
                    args.hard_negatives,
                )
                solution = optimize_ranking_oracle_solution(
                    loaded[name]["train_queries"],
                    loaded[name]["documents"],
                    records,
                    components,
                    steps=args.oracle_steps,
                    learning_rate=args.oracle_learning_rate,
                    penalty=args.oracle_penalty,
                    temperature=args.ranking_temperature,
                    device=args.device,
                )
                oracle_coordinate_sets.append(solution.coordinates)
            actions = fit_candidate_actions(
                oracle_coordinate_sets,
                n_directions=args.n_directions,
                magnitudes=args.magnitudes,
                max_samples_per_corpus=args.max_action_samples_per_corpus,
                random_state=args.random_state,
            )
            reference = fit_reference_geometry(
                {name: loaded[name]["documents"] for name in training},
                n_prototypes=args.n_prototypes,
                max_fit_documents=args.max_fit_documents,
                max_iter=args.prototype_max_iter,
                random_state=args.random_state,
                assignment_batch_size=args.assignment_batch_size,
            )

            quality = {name: {} for name in training}
            for name in training:
                for split in ("train", "dev"):
                    quality[name][split] = _candidate_quality(
                        loaded[name][f"{split}_dataset"],
                        loaded[name][f"{split}_queries"],
                        loaded[name]["documents"],
                        actions,
                        components,
                        args.batch_size,
                    )

            selected_models = {}
            for method in METHODS:
                train_parts = []
                target_parts = []
                sizes = []
                dev_features = {}
                for name in training:
                    geometry = reference if method == "reference_geometry" else local_geometries[name]
                    train_x = _action_features(
                        loaded[name]["train_queries"], actions, components, method, geometry, args
                    )
                    train_y_matrix = quality[name]["train"] - quality[name]["train"][:, [0]]
                    train_parts.append(train_x)
                    target_parts.append(train_y_matrix.reshape(-1))
                    sizes.append(len(train_x))
                    dev_features[name] = _action_features(
                        loaded[name]["dev_queries"], actions, components, method, geometry, args
                    )
                train_x = np.concatenate(train_parts)
                train_y = np.concatenate(target_parts)
                best = None
                for alpha in args.alphas:
                    model = _fit_utility_model(train_x, train_y, sizes, alpha)
                    for threshold in args.thresholds:
                        corpus_scores = []
                        correction_rates = []
                        for name in training:
                            choices = _choose_actions(
                                model.predict(dev_features[name]), len(actions), threshold
                            )
                            corpus_scores.append(
                                float(np.mean(quality[name]["dev"][np.arange(len(choices)), choices]))
                            )
                            correction_rates.append(float(np.mean(choices != 0)))
                        row = {
                            "model": model_key,
                            "heldout_corpus": heldout,
                            "method": method,
                            "alpha": float(alpha),
                            "threshold": float(threshold),
                            "macro_dev_NDCG@10": float(np.mean(corpus_scores)),
                            "macro_dev_correction_rate": float(np.mean(correction_rates)),
                        }
                        selection_rows.append(row)
                        key = (row["macro_dev_NDCG@10"], threshold, -float(alpha))
                        if best is None or key > best[0]:
                            best = (key, model, row)
                selected_models[method] = best

            held = loaded[heldout]
            baseline = evaluate_retrieval(
                held["test_dataset"], held["test_queries"], held["documents"], args.batch_size
            )
            result_rows.append(
                {
                    "model": MODEL_SPECS[model_key]["label"],
                    "heldout_corpus": heldout,
                    "method": "frozen",
                    "selected_alpha": np.nan,
                    "selected_threshold": np.nan,
                    "correction_rate": 0.0,
                    "mean_correction_magnitude": 0.0,
                    **baseline,
                }
            )
            held_choices = {}
            for method, (_key, model, selected) in selected_models.items():
                geometry = reference if method == "reference_geometry" else local_geometries[heldout]
                test_x = _action_features(
                    held["test_queries"], actions, components, method, geometry, args
                )
                choices = _choose_actions(
                    model.predict(test_x), len(actions), selected["threshold"]
                )
                held_choices[method] = choices
                corrected = _selected_queries(held["test_queries"], choices, actions, components)
                metrics = evaluate_retrieval(
                    held["test_dataset"], corrected, held["documents"], args.batch_size
                )
                magnitudes = np.linalg.norm(actions[choices] @ components, axis=1)
                result_rows.append(
                    {
                        "model": MODEL_SPECS[model_key]["label"],
                        "heldout_corpus": heldout,
                        "method": method,
                        "selected_alpha": selected["alpha"],
                        "selected_threshold": selected["threshold"],
                        "correction_rate": float(np.mean(choices != 0)),
                        "mean_correction_magnitude": float(np.mean(magnitudes)),
                        **metrics,
                    }
                )

            held_quality = _candidate_quality(
                held["test_dataset"], held["test_queries"], held["documents"],
                actions, components, args.batch_size,
            )
            oracle_ndcg = float(np.mean(np.max(held_quality, axis=1)))
            for row in result_rows[-(len(METHODS) + 1):]:
                row["candidate_oracle_NDCG@10"] = oracle_ndcg
                row["candidate_oracle_gain"] = oracle_ndcg - baseline["NDCG@10"]
            fold_metadata.append(
                {
                    "model": model_key,
                    "heldout_corpus": heldout,
                    "training_corpora": training,
                    "action_count_including_zero": len(actions),
                    "rank": args.rank,
                    "heldout_qrels_used_for_action_construction": False,
                    "heldout_qrels_used_for_model_or_selection": False,
                    "heldout_documents_used_for_corpus_geometry": True,
                    "selected": {
                        method: {
                            "alpha": values[2]["alpha"],
                            "threshold": values[2]["threshold"],
                            "macro_dev_NDCG@10": values[2]["macro_dev_NDCG@10"],
                        }
                        for method, values in selected_models.items()
                    },
                }
            )
            print(f"{model_key}/{heldout}: complete", flush=True)

    result = pd.DataFrame(result_rows)
    result.to_csv(args.output_dir / "candidate_action_test_metrics.csv", index=False)
    pd.DataFrame(selection_rows).to_csv(
        args.output_dir / "candidate_action_selection.csv", index=False
    )
    (args.output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "protocol": "strict leave-one-corpus-out candidate-action utility scoring",
                "methods": list(METHODS),
                "models": args.models,
                "action_source": "training-only privileged ranking-coordinate direction clusters",
                "utility_target": "per-query NDCG@10 gain over zero action",
                "online_retrievals": 1,
                "folds": fold_metadata,
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
        "--output-dir", type=Path, default=ROOT / "results" / "candidate_action_loco_text_small"
    )
    parser.add_argument(
        "--datasets", nargs="+", default=["fiqa", "arguana", "scifact", "nfcorpus"]
    )
    parser.add_argument(
        "--models", nargs="+", choices=MODEL_SPECS,
        default=["text-embedding-3-small-aiberm"],
    )
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--n-directions", type=int, default=8)
    parser.add_argument("--magnitudes", nargs="+", type=float, default=[0.05, 0.1, 0.2])
    parser.add_argument("--max-action-samples-per-corpus", type=int, default=1000)
    parser.add_argument("--hard-negatives", type=int, default=32)
    parser.add_argument("--ranking-temperature", type=float, default=0.05)
    parser.add_argument("--oracle-steps", type=int, default=100)
    parser.add_argument("--oracle-learning-rate", type=float, default=0.05)
    parser.add_argument("--oracle-penalty", type=float, default=1e-3)
    parser.add_argument("--n-prototypes", type=int, default=32)
    parser.add_argument("--top-m", type=int, default=8)
    parser.add_argument("--geometry-temperature", type=float, default=0.07)
    parser.add_argument("--max-fit-documents", type=int, default=20_000)
    parser.add_argument("--prototype-max-iter", type=int, default=30)
    parser.add_argument("--assignment-batch-size", type=int, default=4096)
    parser.add_argument("--alphas", nargs="+", type=float, default=[1.0, 10.0, 100.0, 1000.0])
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.0, 0.0025, 0.005, 0.01, 0.02])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--random-state", type=int, default=0)
    return parser


if __name__ == "__main__":
    load_dotenv(ROOT / ".env")
    run(build_parser().parse_args())
