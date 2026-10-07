"""Measure corpus-sketch geometry, predictability, and fixed-query shifts."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dense_retrieval.analysis.shared_anchors import SharedAnchorCodebook  # noqa: E402
from dense_retrieval.analysis.study_data import (  # noqa: E402
    other_documents,
    prepare_three_way,
)
from dense_retrieval.embeddings import MODEL_SPECS  # noqa: E402


def sketch(queries, codebook, occupancy):
    features = codebook.query_features(queries, occupancy)
    query_weights = np.exp(
        features.similarities - features.similarities.max(axis=1, keepdims=True)
    )
    query_weights /= np.maximum(query_weights.sum(axis=1, keepdims=True), 1e-12)
    # Reweight the fixed query-side cells by corpus density so fixed-query
    # comparisons expose occupancy changes rather than only anchor geometry.
    weights = query_weights * occupancy.occupancy[features.indices]
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
    values = np.sum(weights[..., None] * codebook.anchors[features.indices], axis=1)
    entropy = -np.sum(weights * np.log(np.maximum(weights, 1e-12)), axis=1)
    return codebook.anchors, values, weights, entropy


def geometry_rows(dataset, model, variant, queries, values, entropy, top_m):
    query_norm = np.linalg.norm(queries, axis=1)
    value_norm = np.linalg.norm(values, axis=1)
    cosine = np.sum(queries * values, axis=1) / np.maximum(query_norm * value_norm, 1e-12)
    distance = np.linalg.norm(values - queries, axis=1)
    rows = {
        "dataset": dataset,
        "model": model,
        "variant": variant,
        "queries": len(queries),
        "cos_q_g_mean": float(np.mean(cosine)),
        "cos_q_g_median": float(np.median(cosine)),
        "cos_q_g_std": float(np.std(cosine)),
        "distance_g_q_mean": float(np.mean(distance)),
        "distance_g_q_median": float(np.median(distance)),
        "distance_g_q_std": float(np.std(distance)),
        "entropy_mean": float(np.mean(entropy)),
        "entropy_median": float(np.median(entropy)),
        "entropy_std": float(np.std(entropy)),
        "entropy_max_log_top_m": float(np.log(top_m)),
    }
    return rows


def stable_masks(ids, fraction=0.2, salt="sketch-ridge-v1"):
    ordered = sorted(ids, key=lambda value: hashlib.sha256(f"{salt}:{value}".encode()).digest())
    count = max(1, int(round(len(ordered) * fraction)))
    validation = set(ordered[:count])
    mask = np.asarray([value in validation for value in ids])
    return ~mask, mask


def prediction_metrics(train_q, train_g, test_q, test_g, train_ids):
    fit_mask, _ = stable_masks(train_ids)
    model = Ridge(alpha=1.0, fit_intercept=True)
    model.fit(train_q[fit_mask], train_g[fit_mask])
    prediction = model.predict(test_q)
    numerator = np.sum(prediction * test_g, axis=1)
    denominator = np.linalg.norm(prediction, axis=1) * np.linalg.norm(test_g, axis=1)
    cosine = numerator / np.maximum(denominator, 1e-12)
    return {
        "r2_variance_weighted": float(r2_score(test_g, prediction, multioutput="variance_weighted")),
        "cos_pred_g_mean": float(np.mean(cosine)),
        "cos_pred_g_median": float(np.median(cosine)),
        "train_rows": int(fit_mask.sum()),
        "test_rows": int(len(test_q)),
    }, model


def load_corpus_data(dataset_name, model_key, cache_root):
    return prepare_three_way(dataset_name, model_key, cache_root)


def fixed_query_variants(
    model_key,
    cache_root,
    n_queries,
    random_state,
    codebook,
    random_external,
    hard_external,
    high_external,
):
    (
        _train_dataset,
        _dev_dataset,
        test_dataset,
        _train_queries,
        _dev_queries,
        test_queries,
        documents,
        _cache_dir,
        _split_method,
    ) = load_corpus_data("fiqa", model_key, cache_root)
    query_ids = [sample.query_id for sample in test_dataset.queries]
    order = np.argsort(
        [hashlib.sha256(f"fixed-query-v1:{qid}".encode()).digest() for qid in query_ids]
    )[:n_queries]
    queries = np.asarray(test_queries)[order]
    external = other_documents("fiqa", model_key, cache_root)
    rng = np.random.default_rng(random_state)
    random_indices = rng.choice(len(external), size=min(random_external, len(external)), replace=False)
    candidate_queries = queries
    scores = candidate_queries @ external.T
    difficulty = np.max(scores, axis=0)
    hard_order = np.argsort(-difficulty, kind="mergesort")
    variants = {
        "D0_original": np.asarray(documents),
        "D1_random_external": np.concatenate(
            [np.asarray(documents), external[random_indices]], axis=0
        ),
        "D2_hard_external": np.concatenate(
            [np.asarray(documents), external[hard_order[: min(hard_external, len(external))]]], axis=0
        ),
        "D3_high_hard_external": np.concatenate(
            [np.asarray(documents), external[hard_order[: min(high_external, len(external))]]], axis=0
        ),
    }
    rows = []
    sketches = {}
    for variant, corpus in variants.items():
        occupancy = codebook.occupancy(corpus)
        prototypes, values, weights, entropy = sketch(queries, codebook, occupancy)
        sketches[variant] = values
        row = geometry_rows(
            "fiqa_fixed_query",
            model_key,
            variant,
            queries,
            values,
            entropy,
            codebook.top_m,
        )
        row["corpus_documents"] = len(corpus)
        row["external_documents"] = len(corpus) - len(documents)
        row["anchor_count"] = len(prototypes)
        rows.append(row)
    base = sketches["D0_original"]
    shift_rows = []
    for variant, values in sketches.items():
        shift = np.linalg.norm(values - base, axis=1)
        cosine = np.sum(values * base, axis=1) / np.maximum(
            np.linalg.norm(values, axis=1) * np.linalg.norm(base, axis=1), 1e-12
        )
        shift_rows.append(
            {
                "model": model_key,
                "variant": variant,
                "cos_g_D0_g_Dv_mean": float(np.mean(cosine)),
                "distance_g_Dv_g_D0_mean": float(np.mean(shift)),
                "distance_g_Dv_g_D0_median": float(np.median(shift)),
            }
        )
    return rows, shift_rows


def run(args):
    geometry = []
    predictability = []
    pooled_predictability = []
    fixed_geometry = []
    fixed_shifts = []
    for model_key in args.models:
        pooled_train_q = []
        pooled_train_g = []
        pooled_test_q = []
        pooled_test_g = []
        pooled_ids = []
        loaded = {}
        all_documents = {}
        for dataset_name in args.datasets:
            (
                train_dataset,
                _dev_dataset,
                test_dataset,
                train_queries,
                _dev_queries,
                test_queries,
                documents,
                _cache_dir,
                _split_method,
            ) = load_corpus_data(dataset_name, model_key, args.cache_root)
            loaded[dataset_name] = (train_dataset, test_dataset, train_queries, test_queries, documents)
            all_documents[dataset_name] = documents
        codebook = SharedAnchorCodebook.fit(
            all_documents,
            n_anchors=args.n_anchors,
            top_m=args.top_m,
            random_state=args.random_state,
        )
        anchor_occupancies = {
            name: codebook.occupancy(values[-1]) for name, values in loaded.items()
        }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        codebook_path = args.output_dir / f"{model_key}_shared_anchors.npz"
        codebook.save(codebook_path, **anchor_occupancies)
        for dataset_name in args.datasets:
            train_dataset, test_dataset, train_queries, test_queries, documents = loaded[dataset_name]
            occupancy = anchor_occupancies[dataset_name]
            _, train_g, _, train_entropy = sketch(train_queries, codebook, occupancy)
            _, test_g, _, test_entropy = sketch(test_queries, codebook, occupancy)
            # Geometry is reported on the held-out test queries.
            geometry.append(
                geometry_rows(
                    dataset_name,
                    model_key,
                    "original",
                    test_queries,
                    test_g,
                    test_entropy,
                    args.top_m,
                )
            )
            train_ids = [f"{dataset_name}:{sample.query_id}" for sample in train_dataset.queries]
            metrics, _model = prediction_metrics(
                train_queries, train_g, test_queries, test_g, train_ids
            )
            predictability.append({"dataset": dataset_name, "model": model_key, **metrics})
            pooled_train_q.append(train_queries)
            pooled_train_g.append(train_g)
            pooled_test_q.append(test_queries)
            pooled_test_g.append(test_g)
            pooled_ids.extend(train_ids)
        pooled_metrics, _ = prediction_metrics(
            np.concatenate(pooled_train_q),
            np.concatenate(pooled_train_g),
            np.concatenate(pooled_test_q),
            np.concatenate(pooled_test_g),
            pooled_ids,
        )
        pooled_predictability.append({"model": model_key, **pooled_metrics})
        rows, shifts = fixed_query_variants(
            model_key,
            args.cache_root,
            args.fixed_queries,
            args.random_state,
            codebook,
            args.random_external,
            args.hard_external,
            args.high_external,
        )
        fixed_geometry.extend(rows)
        fixed_shifts.extend(shifts)
        print(f"{model_key}: complete", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(geometry).to_csv(args.output_dir / "sketch_geometry.csv", index=False)
    pd.DataFrame(predictability).to_csv(args.output_dir / "sketch_predictability_by_corpus.csv", index=False)
    pd.DataFrame(pooled_predictability).to_csv(args.output_dir / "sketch_predictability_pooled.csv", index=False)
    pd.DataFrame(fixed_geometry).to_csv(args.output_dir / "fixed_query_geometry.csv", index=False)
    pd.DataFrame(fixed_shifts).to_csv(args.output_dir / "fixed_query_sketch_shifts.csv", index=False)
    (args.output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "datasets": args.datasets,
                "models": args.models,
                "n_anchors": args.n_anchors,
                "top_m": args.top_m,
                "fixed_query": {
                    "base_corpus": "fiqa",
                    "query_count": args.fixed_queries,
                    "random_external": args.random_external,
                    "hard_external": args.hard_external,
                    "high_external": args.high_external,
                },
                "ridge_alpha": 1.0,
                "r2_is_held_out": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=ROOT / "cache")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "corpus_sketch_analysis")
    parser.add_argument("--datasets", nargs="+", default=["fiqa", "arguana", "scifact", "nfcorpus"])
    parser.add_argument("--models", nargs="+", choices=MODEL_SPECS, default=list(MODEL_SPECS))
    parser.add_argument("--n-anchors", type=int, default=256)
    parser.add_argument("--top-m", type=int, default=16)
    parser.add_argument("--fixed-queries", type=int, default=64)
    parser.add_argument("--random-external", type=int, default=1000)
    parser.add_argument("--hard-external", type=int, default=1000)
    parser.add_argument("--high-external", type=int, default=5000)
    parser.add_argument("--random-state", type=int, default=0)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
