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
import torch
from dotenv import load_dotenv
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dense_retrieval.analysis.prototype_correction import (  # noqa: E402
    project_correction_targets,
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
from dense_retrieval.retrieval.exact import exact_search  # noqa: E402


METHODS = ["baseline", "rank_q_only", "rank_corpus_sketch"]


class CorrectionMLP(nn.Module):
    def __init__(self, input_dimension: int, rank: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dimension, 256),
            nn.ReLU(),
            nn.Linear(256, rank),
        )

    def forward(self, values):
        return self.network(values)


def corpus_sketch(queries, codebook, occupancy):
    """Return flattened Top-M per-anchor query-relative corpus features.

    Unlike a pooled anchor vector, this preserves which fixed semantic cells
    are dense in the current corpus. ``codebook.projection`` is the shared
    B128 basis, so each row contains ``[B.T(u_k-q), q.u_k, log(p_D,k)]``.
    """
    features = codebook.query_features(queries, occupancy)
    return features.values.reshape(len(features.values), -1).astype(np.float32)


def fit_feature_normalizer(values):
    """Fit per-column robust scaling using training sketches only."""
    values = np.asarray(values, dtype=np.float64)
    center = np.median(values, axis=0)
    scale = np.median(np.abs(values - center), axis=0) * 1.4826
    scale = np.where(scale > 1e-6, scale, 1.0)
    return center.astype(np.float32), scale.astype(np.float32)


def normalize_features(values, center, scale):
    return np.clip(
        (np.asarray(values, dtype=np.float32) - center) / scale, -8.0, 8.0
    ).astype(np.float32)


def build_rank_records(dataset, query_embeddings, documents, hard_k):
    """Build positive/all-hard-negative index arrays for one corpus."""
    document_index = {str(doc_id): row for row, doc_id in enumerate(dataset.corpus)}
    positives = []
    negatives = []
    kept_rows = []
    hard_indices, _ = exact_search(query_embeddings, documents, min(hard_k + 32, len(documents)))
    for row, sample in enumerate(dataset.queries):
        positive = [
            document_index[str(doc_id)]
            for doc_id, relevance in dataset.qrels[str(sample.query_id)].items()
            if int(relevance) > 0 and str(doc_id) in document_index
        ]
        if not positive:
            continue
        positive_set = set(positive)
        negative = [int(index) for index in hard_indices[row] if int(index) not in positive_set]
        if not negative:
            continue
        kept_rows.append(row)
        positives.append(positive)
        negatives.append(negative[:hard_k])
    if not kept_rows:
        raise ValueError(f"{dataset.name}: no train queries with positives and negatives")
    max_positive = max(map(len, positives))
    max_negative = max(map(len, negatives))
    positive_array = np.full((len(kept_rows), max_positive), -1, dtype=np.int64)
    negative_array = np.full((len(kept_rows), max_negative), -1, dtype=np.int64)
    for row, values in enumerate(positives):
        positive_array[row, : len(values)] = values
    for row, values in enumerate(negatives):
        negative_array[row, : len(values)] = values
    return {
        "query_rows": np.asarray(kept_rows, dtype=np.int64),
        "positive_indices": positive_array,
        "negative_indices": negative_array,
    }


def train_rank_model(
    corpora,
    mean,
    components,
    *,
    use_sketch,
    epochs,
    batch_size,
    train_lambdas,
    temperature,
    learning_rate,
    weight_decay,
    random_state,
    hard_negatives,
    device="auto",
):
    torch.manual_seed(random_state)
    device = torch.device(
        "cuda" if device == "auto" and torch.cuda.is_available() else device
        if device != "auto" else "cpu"
    )
    dimension = next(iter(corpora.values()))["train_queries"].shape[1]
    rank = components.shape[0]
    sketch_dimension = next(iter(corpora.values()))["train_sketches"].shape[1]
    model = CorrectionMLP(dimension + sketch_dimension, rank).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    mean_tensor = torch.from_numpy(mean.astype(np.float32)).to(device)
    components_tensor = torch.from_numpy(components.astype(np.float32)).to(device)
    losses = []
    corpus_items = list(corpora.items())
    for epoch in range(epochs):
        epoch_losses = []
        # Refresh the competition set with the current model.  This prevents the
        # predictor from only learning to avoid a stale first-pass top-k list.
        for corpus_index, (name, data) in enumerate(corpus_items):
            q = data["train_queries"]
            sketch = data["train_sketches"]
            features = np.concatenate(
                [q, sketch if use_sketch else np.zeros_like(sketch)], axis=1
            )
            with torch.no_grad():
                coordinates = model(torch.from_numpy(features.astype(np.float32)).to(device))
                deltas = mean_tensor + coordinates @ components_tensor
            probe = _apply(q, deltas.cpu().numpy(), max(train_lambdas))
            data["rank_records"] = build_rank_records(
                data["train_dataset"], probe, data["documents"], hard_negatives
            )

        # Round-robin batches give each corpus the same number of optimizer
        # updates per epoch; smaller corpora are cycled with replacement.
        batch_orders = []
        max_batches = 0
        for corpus_index, (_, data) in enumerate(corpus_items):
            rng = np.random.default_rng(random_state + epoch * 1009 + corpus_index)
            records = data["rank_records"]
            order = rng.permutation(len(records["query_rows"]))
            batch_orders.append(order)
            max_batches = max(max_batches, int(np.ceil(len(order) / batch_size)))
        for batch_id in range(max_batches):
            for corpus_index, (_, data) in enumerate(corpus_items):
                records = data["rank_records"]
                order = batch_orders[corpus_index]
                if len(order) == 0:
                    continue
                positions = (np.arange(batch_size) + batch_id * batch_size) % len(order)
                selected = order[positions]
                q = data["train_queries"][records["query_rows"][selected]]
                sketch = data["train_sketches"][records["query_rows"][selected]]
                features = np.concatenate(
                    [q, sketch if use_sketch else np.zeros_like(sketch)], axis=1
                )
                q_tensor = torch.from_numpy(q.astype(np.float32)).to(device)
                features_tensor = torch.from_numpy(features.astype(np.float32)).to(device)
                positive_indices = records["positive_indices"][selected]
                negative_indices = records["negative_indices"][selected]
                positive_docs = torch.from_numpy(
                    data["documents"][positive_indices].astype(np.float32)
                ).to(device)
                negative_docs = torch.from_numpy(
                    data["documents"][negative_indices].astype(np.float32)
                ).to(device)
                coordinates = model(features_tensor)
                deltas = mean_tensor + coordinates @ components_tensor
                lambda_ = float(rng.choice(train_lambdas))
                corrected = q_tensor + lambda_ * deltas
                corrected = F.normalize(corrected, dim=1)
                positive_scores = torch.einsum("bd,bpd->bp", corrected, positive_docs)
                negative_scores = torch.einsum("bd,bnd->bn", corrected, negative_docs)
                loss = multi_positive_hard_negative_loss(
                    positive_scores,
                    negative_scores,
                    torch.from_numpy(positive_indices >= 0).to(device),
                    torch.from_numpy(negative_indices >= 0).to(device),
                    temperature=temperature,
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_losses.append(float(loss.detach()))
        losses.append(float(np.mean(epoch_losses)))
    return model, losses


def multi_positive_hard_negative_loss(
    positive_scores,
    negative_scores,
    positive_mask,
    negative_mask,
    *,
    temperature: float = 0.05,
):
    """Compute a temperature-scaled masked multi-positive log-softmax loss."""
    if temperature <= 0 or not np.isfinite(temperature):
        raise ValueError("temperature must be finite and positive")
    positive_scores = positive_scores / float(temperature)
    negative_scores = negative_scores / float(temperature)
    positive_scores = positive_scores.masked_fill(~positive_mask, -torch.inf)
    negative_scores = negative_scores.masked_fill(~negative_mask, -torch.inf)
    numerator = torch.logsumexp(positive_scores, dim=1)
    denominator = torch.logsumexp(
        torch.cat([positive_scores, negative_scores], dim=1), dim=1
    )
    return -(numerator - denominator).mean()


def predict_model(model, queries, sketches, mean, components, use_sketch):
    features = np.concatenate(
        [queries, sketches if use_sketch else np.zeros_like(sketches)], axis=1
    )
    device = next(model.parameters()).device
    with torch.no_grad():
        coordinates = model(torch.from_numpy(features.astype(np.float32)).to(device)).cpu().numpy()
    return mean + coordinates @ components


def select_lambda(corpora, predictor, lambdas, batch_size):
    rows = []
    for lambda_ in lambdas:
        values = []
        for name, data in corpora.items():
            metrics = evaluate_retrieval(
                data["dev_dataset"],
                _apply(data["dev_queries"], predictor(name, "dev"), lambda_),
                data["documents"],
                batch_size,
            )
            values.append(metrics["HitRate@10"])
            rows.append({"dataset": name, "lambda": lambda_, **metrics})
        rows.append(
            {"dataset": "macro", "lambda": lambda_, "HitRate@10": float(np.mean(values))}
        )
    frame = pd.DataFrame(rows)
    macro = frame[frame.dataset == "macro"]
    selected = float(macro.loc[macro["HitRate@10"].idxmax(), "lambda"])
    return selected, frame


def _apply(queries, deltas, lambda_):
    corrected = np.asarray(queries, dtype=np.float64) + float(lambda_) * deltas
    corrected /= np.maximum(np.linalg.norm(corrected, axis=1, keepdims=True), 1e-12)
    return corrected.astype(np.float32)


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
