"""Train corpus-conditioned query correction with a retrieval ranking loss.

The shared 128-D coordinate system is fitted once from pooled train deltas.
The correction predictor is then optimized directly against frozen-corpus
positive and hard-negative scores.  Hard negatives are generated separately
inside each corpus, so the supervision reflects that corpus's competition
structure rather than only the positive embedding.
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

from dense_retrieval.analysis.ranking_correction import (  # noqa: E402
    _apply,
    build_rank_records,
    corpus_sketch,
    fit_feature_normalizer,
    multi_positive_hard_negative_loss,
    normalize_features,
    predict_model,
    select_lambda,
    train_rank_model,
)
from dense_retrieval.analysis.shared_anchors import SharedAnchorCodebook  # noqa: E402
from dense_retrieval.analysis.study_data import (  # noqa: E402
    DEFAULT_LAMBDAS,
    prepare_three_way,
)
from dense_retrieval.analysis.query_correction import (  # noqa: E402
    build_oracle_deltas,
    fit_zero_origin_pca_subspace,
)
from dense_retrieval.embeddings import MODEL_SPECS  # noqa: E402
from dense_retrieval.evaluation.ranking import evaluate_retrieval  # noqa: E402


METHODS = ["baseline", "rank_q_only", "rank_corpus_sketch"]


def run(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_rows = []
    dev_rows = []
    metadata = []
    for model_key in args.models:
        corpora = {}
        pooled_deltas = []
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
        for dataset_name in args.datasets:
            data = loaded[dataset_name]
            train_dataset = data["train_dataset"]
            train_queries = data["train_queries"]
            documents = data["documents"]
            train_rows = data["train_rows"]
            train_deltas = data["train_deltas"]
            train_missing = data["train_missing"]
            rank_records = build_rank_records(
                train_dataset, train_queries, documents, args.hard_negatives
            )
            corpora[dataset_name] = {
                "train_dataset": train_dataset,
                "dev_dataset": data["dev_dataset"],
                "test_dataset": data["test_dataset"],
                "train_queries": train_queries,
                "dev_queries": data["dev_queries"],
                "test_queries": data["test_queries"],
                "documents": documents,
                "rank_records": rank_records,
                "cache_dir": data["cache_dir"],
                "split_method": data["split_method"],
                "train_missing": train_missing,
            }
            pooled_deltas.append(train_deltas)
        mean, components = fit_zero_origin_pca_subspace(
            np.concatenate(pooled_deltas), args.rank
        )
        # Build query-relative per-anchor features in the shared B128 basis.
        projected_codebook = SharedAnchorCodebook(
            codebook.anchors, top_m=args.top_m, projection=components
        )
        anchor_occupancies = {
            name: projected_codebook.occupancy(data["documents"])
            for name, data in loaded.items()
        }
        codebook_path = args.output_dir / "models" / f"{model_key}_shared_anchors.npz"
        codebook_path.parent.mkdir(parents=True, exist_ok=True)
        projected_codebook.save(codebook_path, **anchor_occupancies)
        for dataset_name in args.datasets:
            data = loaded[dataset_name]
            occupancy = anchor_occupancies[dataset_name]
            sketches = {
                "train": corpus_sketch(data["train_queries"], projected_codebook, occupancy),
                "dev": corpus_sketch(data["dev_queries"], projected_codebook, occupancy),
                "test": corpus_sketch(data["test_queries"], projected_codebook, occupancy),
            }
            corpora[dataset_name]["sketches"] = sketches
            corpora[dataset_name]["train_sketches"] = sketches["train"]
        all_sketches = np.concatenate(
            [corpora[name]["train_sketches"] for name in args.datasets]
        )
        sketch_center, sketch_scale = fit_feature_normalizer(all_sketches)
        for dataset_name in args.datasets:
            corpora[dataset_name]["sketches"] = {
                split: normalize_features(values, sketch_center, sketch_scale)
                for split, values in corpora[dataset_name]["sketches"].items()
            }
            corpora[dataset_name]["train_sketches"] = corpora[dataset_name]["sketches"]["train"]
        models = {}
        loss_history = {}
        for method, use_sketch in (("rank_q_only", False), ("rank_corpus_sketch", True)):
            model, losses = train_rank_model(
                corpora,
                mean,
                components,
                use_sketch=use_sketch,
                epochs=args.epochs,
                batch_size=args.batch_size,
                train_lambdas=args.train_lambdas,
                temperature=args.temperature,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                random_state=args.random_state,
                hard_negatives=args.hard_negatives,
                device=args.device,
            )
            models[method] = (model, use_sketch)
            loss_history[method] = losses

        predictors = {
            method: lambda name, split, model=model, use_sketch=use_sketch: predict_model(
                model,
                corpora[name][f"{split}_queries"],
                corpora[name]["sketches"][split],
                mean,
                components,
                use_sketch,
            )
            for method, (model, use_sketch) in models.items()
        }
        selected = {"baseline": 0.0}
        for method, predictor in predictors.items():
            lambda_, curve = select_lambda(corpora, predictor, args.lambdas, args.batch_size)
            selected[method] = lambda_
            curve.insert(0, "model", model_key)
            curve.insert(0, "method", method)
            dev_rows.extend(curve.to_dict("records"))
        for name, data in corpora.items():
            baseline = evaluate_retrieval(
                data["test_dataset"], data["test_queries"], data["documents"], args.batch_size
            )
            output_rows.append(
                {"dataset": name, "model": model_key, "method": "baseline", "selected_lambda": 0.0, **baseline}
            )
            for method, predictor in predictors.items():
                metrics = evaluate_retrieval(
                    data["test_dataset"],
                    _apply(data["test_queries"], predictor(name, "test"), selected[method]),
                    data["documents"],
                    args.batch_size,
                )
                output_rows.append(
                    {
                        "dataset": name,
                        "model": model_key,
                        "method": method,
                        "selected_lambda": selected[method],
                        **metrics,
                    }
                )
        metadata.append(
            {
                "model": model_key,
                "datasets": list(args.datasets),
                "shared_coordinate_system": True,
                "pooled_global_mu_B_fit_once": True,
                "zero_origin_correction_subspace": True,
                "rank": args.rank,
                "n_anchors_global": args.n_anchors,
                "top_m": args.top_m,
                "per_anchor_features": ["B.T(anchor-query)", "query-anchor-similarity", "log_occupancy"],
                "per_anchor_feature_width": int(args.rank + 2),
                "sketch_normalization": "training_median_mad_clip8",
                "dynamic_hard_negative_refresh": "every_epoch_after_current_model",
                "corpus_balanced_optimizer_schedule": "round_robin_equal_updates_with_replacement",
                "device": str(next(iter(models.values()))[0].parameters().__next__().device),
                "anchor_codebook_fitted_once_per_embedding_model": True,
                "anchor_codebook": str(codebook_path.relative_to(args.output_dir)),
                "hard_negatives_per_query": args.hard_negatives,
                "ranking_temperature": args.temperature,
                "ranking_train_lambdas": args.train_lambdas,
                "epochs": args.epochs,
                "loss_history": loss_history,
                "selected_lambdas_from_macro_dev_HitRate@10": selected,
                "test_qrels_used_for_training_or_selection": False,
                "corpora": {
                    name: {
                        "split_method": data["split_method"],
                        "train_queries": len(data["train_queries"]),
                        "rank_train_queries": len(data["rank_records"]["query_rows"]),
                        "dev_queries": len(data["dev_queries"]),
                        "test_queries": len(data["test_queries"]),
                        "train_missing_positive": data["train_missing"],
                    }
                    for name, data in corpora.items()
                },
            }
        )
        print(f"{model_key}: complete", flush=True)
    pd.DataFrame(output_rows).to_csv(args.output_dir / "rank_conditioned_test_metrics.csv", index=False)
    pd.DataFrame(dev_rows).to_csv(args.output_dir / "rank_conditioned_dev_curves.csv", index=False)
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=ROOT / "cache")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "rank_conditioned_correction")
    parser.add_argument("--datasets", nargs="+", default=["fiqa", "arguana", "scifact", "nfcorpus"])
    parser.add_argument("--models", nargs="+", choices=MODEL_SPECS, default=list(MODEL_SPECS))
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--n-anchors", type=int, default=256)
    parser.add_argument("--top-m", type=int, default=16)
    parser.add_argument("--hard-negatives", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument(
        "--train-lambdas",
        nargs="+",
        type=float,
        default=[0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0],
    )
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--lambdas", nargs="+", type=float, default=DEFAULT_LAMBDAS)
    return parser


if __name__ == "__main__":
    load_dotenv(ROOT / ".env")
    run(build_parser().parse_args())
