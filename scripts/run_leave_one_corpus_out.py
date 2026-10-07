"""Evaluate shared correction under leave-one-corpus-out transfer.

For each held-out corpus, anchors, the zero-origin correction basis, predictor,
feature scaling, and lambda are fitted using the other corpora only.  The held-
out corpus contributes frozen documents and query embeddings for occupancy and
one final retrieval evaluation.
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

from dense_retrieval.analysis.query_correction import (  # noqa: E402
    build_oracle_deltas,
    fit_zero_origin_pca_subspace,
)
from dense_retrieval.analysis.ranking_correction import (  # noqa: E402
    _apply,
    build_rank_records,
    corpus_sketch,
    fit_feature_normalizer,
    normalize_features,
    predict_model,
    train_rank_model,
)
from dense_retrieval.analysis.shared_anchors import SharedAnchorCodebook  # noqa: E402
from dense_retrieval.analysis.study_data import DEFAULT_LAMBDAS, prepare_three_way  # noqa: E402
from dense_retrieval.embeddings import MODEL_SPECS  # noqa: E402
from dense_retrieval.evaluation.ranking import evaluate_retrieval  # noqa: E402


def run(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    metadata = []
    for model_key in args.models:
        for heldout in args.datasets:
            train_names = [name for name in args.datasets if name != heldout]
            loaded = {}
            all_documents = {}
            for name in args.datasets:
                values = prepare_three_way(name, model_key, args.cache_root)
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
                ) = values
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
                    "cache_dir": cache_dir,
                    "split_method": split_method,
                }
                all_documents[name] = documents

            codebook = SharedAnchorCodebook.fit(
                {name: all_documents[name] for name in train_names},
                n_anchors=args.n_anchors,
                top_m=args.top_m,
                random_state=args.random_state,
            )
            pooled_deltas = np.concatenate([loaded[name]["train_deltas"] for name in train_names])
            mean, components = fit_zero_origin_pca_subspace(pooled_deltas, args.rank)
            projected = SharedAnchorCodebook(
                codebook.anchors, top_m=args.top_m, projection=components
            )
            occupancies = {
                name: projected.occupancy(all_documents[name]) for name in args.datasets
            }
            corpora = {}
            for name in train_names:
                data = loaded[name]
                records = build_rank_records(
                    data["train_dataset"], data["train_queries"], data["documents"], args.hard_negatives
                )
                sketches = {
                    split: corpus_sketch(data[f"{split}_queries"], projected, occupancies[name])
                    for split in ("train", "dev", "test")
                }
                corpora[name] = {
                    **data,
                    "rank_records": records,
                    "sketches": sketches,
                    "train_sketches": sketches["train"],
                }
            held_data = loaded[heldout]
            held_sketches = {
                split: corpus_sketch(held_data[f"{split}_queries"], projected, occupancies[heldout])
                for split in ("train", "dev", "test")
            }
            center, scale = fit_feature_normalizer(
                np.concatenate([corpora[name]["train_sketches"] for name in train_names])
            )
            for name in train_names:
                corpora[name]["sketches"] = {
                    split: normalize_features(value, center, scale)
                    for split, value in corpora[name]["sketches"].items()
                }
                corpora[name]["train_sketches"] = corpora[name]["sketches"]["train"]
            held_sketches = {
                split: normalize_features(value, center, scale)
                for split, value in held_sketches.items()
            }

            models = {}
            loss_history = {}
            for method, use_sketch in (("rank_q_only", False), ("rank_corpus_sketch", True)):
                model, losses = train_rank_model(
                    corpora, mean, components, use_sketch=use_sketch,
                    epochs=args.epochs, batch_size=args.batch_size,
                    train_lambdas=args.train_lambdas, temperature=args.temperature,
                    learning_rate=args.learning_rate, weight_decay=args.weight_decay,
                    random_state=args.random_state, hard_negatives=args.hard_negatives,
                    device=args.device,
                )
                models[method] = (model, use_sketch)
                loss_history[method] = losses

            selected = {}
            for method, (model, use_sketch) in models.items():
                curve = []
                for lambda_ in args.lambdas:
                    values = []
                    for name in train_names:
                        data = corpora[name]
                        deltas = predict_model(
                            model, data["dev_queries"], data["sketches"]["dev"],
                            mean, components, use_sketch
                        )
                        metrics = evaluate_retrieval(
                            data["dev_dataset"], _apply(data["dev_queries"], deltas, lambda_),
                            data["documents"], args.batch_size
                        )
                        values.append(metrics["HitRate@10"])
                    curve.append((float(lambda_), float(np.mean(values))))
                selected[method] = max(curve, key=lambda value: (value[1], -value[0]))[0]

            baseline = evaluate_retrieval(
                held_data["test_dataset"], held_data["test_queries"], held_data["documents"], args.batch_size
            )
            rows.append({"dataset": heldout, "model": model_key, "method": "baseline", "selected_lambda": 0.0, **baseline})
            for method, (model, use_sketch) in models.items():
                deltas = predict_model(
                    model, held_data["test_queries"], held_sketches["test"], mean, components, use_sketch
                )
                metrics = evaluate_retrieval(
                    held_data["test_dataset"],
                    _apply(held_data["test_queries"], deltas, selected[method]),
                    held_data["documents"], args.batch_size,
                )
                rows.append({
                    "dataset": heldout, "model": model_key, "method": method,
                    "selected_lambda": selected[method], **metrics,
                })
            metadata.append({
                "model": model_key,
                "heldout_corpus": heldout,
                "training_corpora": train_names,
                "zero_origin_correction_subspace": True,
                "anchors_fit_on_training_corpora_only": True,
                "sketch_normalization": "training_median_mad_clip8",
                "dynamic_hard_negative_refresh": "every_epoch_after_current_model",
                "corpus_balanced_optimizer_schedule": "round_robin_equal_updates_with_replacement",
                "selected_lambdas_from_training_corpora_dev_HitRate@10": selected,
                "loss_history": loss_history,
                "test_qrels_used_for_training_or_selection": False,
            })
            print(f"{model_key}/{heldout}: complete", flush=True)
    pd.DataFrame(rows).to_csv(args.output_dir / "leave_one_corpus_out_test_metrics.csv", index=False)
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=ROOT / "cache")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "leave_one_corpus_out")
    parser.add_argument("--datasets", nargs="+", default=["fiqa", "arguana", "scifact", "nfcorpus"])
    parser.add_argument("--models", nargs="+", choices=MODEL_SPECS, default=["text-embedding-3-small-aiberm"])
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--n-anchors", type=int, default=256)
    parser.add_argument("--top-m", type=int, default=16)
    parser.add_argument("--hard-negatives", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--train-lambdas", nargs="+", type=float, default=[0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0])
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
