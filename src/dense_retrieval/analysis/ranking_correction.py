"""Shared ranking-conditioned query correction primitives.

This module contains the reusable trainer and feature construction used by the
regular shared-corpus run and the leave-one-corpus-out transfer run.  CLI
scripts should only assemble datasets and call these functions.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from ..evaluation.ranking import evaluate_retrieval
from ..retrieval.exact import exact_search


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
    """Return flattened Top-M query-relative anchor/corpus features."""
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
    hard_indices, _ = exact_search(
        query_embeddings, documents, min(hard_k + 32, len(documents))
    )
    for row, sample in enumerate(dataset.queries):
        positive = [
            document_index[str(doc_id)]
            for doc_id, relevance in dataset.qrels[str(sample.query_id)].items()
            if int(relevance) > 0 and str(doc_id) in document_index
        ]
        if not positive:
            continue
        positive_set = set(positive)
        negative = [
            int(index)
            for index in hard_indices[row]
            if int(index) not in positive_set
        ]
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


def _resolve_device(device):
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be one of: auto, cpu, cuda")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return torch.device("cuda" if device == "auto" and torch.cuda.is_available() else device if device != "auto" else "cpu")


def train_rank_model(
    corpora: Mapping,
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
    resolved_device = _resolve_device(device)
    dimension = next(iter(corpora.values()))["train_queries"].shape[1]
    rank = components.shape[0]
    sketch_dimension = next(iter(corpora.values()))["train_sketches"].shape[1]
    model = CorrectionMLP(dimension + sketch_dimension, rank).to(resolved_device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    mean_tensor = torch.from_numpy(mean.astype(np.float32)).to(resolved_device)
    components_tensor = torch.from_numpy(components.astype(np.float32)).to(resolved_device)
    losses = []
    corpus_items = list(corpora.items())
    for epoch in range(epochs):
        epoch_losses = []
        for corpus_index, (_, data) in enumerate(corpus_items):
            q = data["train_queries"]
            sketch = data["train_sketches"]
            features = np.concatenate(
                [q, sketch if use_sketch else np.zeros_like(sketch)], axis=1
            )
            with torch.no_grad():
                coordinates = model(
                    torch.from_numpy(features.astype(np.float32)).to(resolved_device)
                )
                deltas = mean_tensor + coordinates @ components_tensor
            probe = _apply(q, deltas.cpu().numpy(), max(train_lambdas))
            data["rank_records"] = build_rank_records(
                data["train_dataset"], probe, data["documents"], hard_negatives
            )

        batch_orders = []
        batch_rngs = []
        max_batches = 0
        for corpus_index, (_, data) in enumerate(corpus_items):
            rng = np.random.default_rng(random_state + epoch * 1009 + corpus_index)
            order = rng.permutation(len(data["rank_records"]["query_rows"]))
            batch_orders.append(order)
            batch_rngs.append(rng)
            max_batches = max(max_batches, int(np.ceil(len(order) / batch_size)))
        for batch_id in range(max_batches):
            for corpus_index, (_, data) in enumerate(corpus_items):
                records = data["rank_records"]
                order = batch_orders[corpus_index]
                if len(order) == 0:
                    continue
                rng = batch_rngs[corpus_index]
                positions = (np.arange(batch_size) + batch_id * batch_size) % len(order)
                selected = order[positions]
                q = data["train_queries"][records["query_rows"][selected]]
                sketch = data["train_sketches"][records["query_rows"][selected]]
                features = np.concatenate(
                    [q, sketch if use_sketch else np.zeros_like(sketch)], axis=1
                )
                q_tensor = torch.from_numpy(q.astype(np.float32)).to(resolved_device)
                features_tensor = torch.from_numpy(features.astype(np.float32)).to(resolved_device)
                positive_indices = records["positive_indices"][selected]
                negative_indices = records["negative_indices"][selected]
                positive_docs = torch.from_numpy(
                    data["documents"][positive_indices].astype(np.float32)
                ).to(resolved_device)
                negative_docs = torch.from_numpy(
                    data["documents"][negative_indices].astype(np.float32)
                ).to(resolved_device)
                coordinates = model(features_tensor)
                deltas = mean_tensor + coordinates @ components_tensor
                lambda_ = float(rng.choice(train_lambdas))
                corrected = F.normalize(q_tensor + lambda_ * deltas, dim=1)
                positive_scores = torch.einsum("bd,bpd->bp", corrected, positive_docs)
                negative_scores = torch.einsum("bd,bnd->bn", corrected, negative_docs)
                loss = multi_positive_hard_negative_loss(
                    positive_scores,
                    negative_scores,
                    torch.from_numpy(positive_indices >= 0).to(resolved_device),
                    torch.from_numpy(negative_indices >= 0).to(resolved_device),
                    temperature=temperature,
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_losses.append(float(loss.detach().cpu()))
        losses.append(float(np.mean(epoch_losses)))
    return model, losses


def multi_positive_hard_negative_loss(
    positive_scores, negative_scores, positive_mask, negative_mask, *, temperature=0.05
):
    if temperature <= 0 or not np.isfinite(temperature):
        raise ValueError("temperature must be finite and positive")
    positive_scores = (positive_scores / float(temperature)).masked_fill(
        ~positive_mask, -torch.inf
    )
    negative_scores = (negative_scores / float(temperature)).masked_fill(
        ~negative_mask, -torch.inf
    )
    numerator = torch.logsumexp(positive_scores, dim=1)
    denominator = torch.logsumexp(torch.cat([positive_scores, negative_scores], dim=1), dim=1)
    return -(numerator - denominator).mean()


def predict_model(model, queries, sketches, mean, components, use_sketch):
    features = np.concatenate(
        [queries, sketches if use_sketch else np.zeros_like(sketches)], axis=1
    )
    device = next(model.parameters()).device
    with torch.no_grad():
        coordinates = model(
            torch.from_numpy(features.astype(np.float32)).to(device)
        ).cpu().numpy()
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
        rows.append({"dataset": "macro", "lambda": lambda_, "HitRate@10": float(np.mean(values))})
    frame = pd.DataFrame(rows)
    macro = frame[frame.dataset == "macro"]
    selected = float(macro.loc[macro["HitRate@10"].idxmax(), "lambda"])
    return selected, frame


def _apply(queries, deltas, lambda_):
    corrected = np.asarray(queries, dtype=np.float64) + float(lambda_) * deltas
    corrected /= np.maximum(np.linalg.norm(corrected, axis=1, keepdims=True), 1e-12)
    return corrected.astype(np.float32)


__all__ = [
    "CorrectionMLP",
    "corpus_sketch",
    "fit_feature_normalizer",
    "normalize_features",
    "build_rank_records",
    "train_rank_model",
    "multi_positive_hard_negative_loss",
    "predict_model",
    "select_lambda",
    "_apply",
]
