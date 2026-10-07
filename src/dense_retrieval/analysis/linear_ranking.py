"""Linear ranking-aligned correction in a frozen low-rank query subspace."""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .ranking_correction import (
    _resolve_device,
    build_rank_records,
    multi_positive_hard_negative_loss,
)


class LinearRankCorrection(nn.Module):
    """A zero-origin linear map from query features to correction coordinates."""

    def __init__(self, input_dimension: int, rank: int):
        super().__init__()
        self.linear = nn.Linear(input_dimension, rank)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, values):
        return self.linear(values)


def _document_batch(documents, indices, device):
    return torch.from_numpy(
        np.asarray(documents[np.maximum(indices, 0)], dtype=np.float32)
    ).to(device)


def train_linear_rank_model(
    corpora: Mapping,
    components: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    train_lambdas: list[float],
    temperature: float,
    learning_rate: float,
    weight_decay: float,
    coordinate_penalty: float,
    hard_negatives: int,
    random_state: int,
    device: str = "auto",
) -> tuple[LinearRankCorrection, list[float]]:
    """Fit one corpus-balanced linear predictor with static baseline negatives."""
    if not corpora:
        raise ValueError("corpora must not be empty")
    if epochs < 1 or batch_size < 1 or hard_negatives < 1:
        raise ValueError("epochs, batch_size, and hard_negatives must be positive")
    if not train_lambdas or any(value <= 0 for value in train_lambdas):
        raise ValueError("train_lambdas must contain positive values")
    if coordinate_penalty < 0:
        raise ValueError("coordinate_penalty must be non-negative")

    torch.manual_seed(random_state)
    resolved_device = _resolve_device(device)
    items = list(corpora.items())
    input_dimension = items[0][1]["train_features"].shape[1]
    rank = np.asarray(components).shape[0]
    if any(data["train_features"].shape[1] != input_dimension for _, data in items):
        raise ValueError("all corpora must use the same feature dimension")

    model = LinearRankCorrection(input_dimension, rank).to(resolved_device)
    optimizer = torch.optim.AdamW(
        model.parameters(), learning_rate, weight_decay=weight_decay
    )
    basis = torch.from_numpy(np.asarray(components, dtype=np.float32)).to(
        resolved_device
    )
    for _, data in items:
        data["rank_records"] = build_rank_records(
            data["train_dataset"],
            data["train_queries"],
            data["documents"],
            hard_negatives,
        )

    history = []
    for epoch in range(int(epochs)):
        orders = []
        rngs = []
        max_batches = 0
        for corpus_index, (_, data) in enumerate(items):
            records = data["rank_records"]
            rng = np.random.default_rng(
                random_state + epoch * 1009 + corpus_index
            )
            order = rng.permutation(len(records["query_rows"]))
            orders.append(order)
            rngs.append(rng)
            max_batches = max(max_batches, int(np.ceil(len(order) / batch_size)))

        epoch_losses = []
        for batch_id in range(max_batches):
            for corpus_index, (_, data) in enumerate(items):
                records = data["rank_records"]
                order = orders[corpus_index]
                positions = (
                    np.arange(batch_size) + batch_id * batch_size
                ) % len(order)
                selected = order[positions]
                query_rows = records["query_rows"][selected]
                positive_indices = records["positive_indices"][selected]
                negative_indices = records["negative_indices"][selected]
                queries = torch.from_numpy(
                    np.asarray(data["train_queries"][query_rows], dtype=np.float32)
                ).to(resolved_device)
                features = torch.from_numpy(
                    np.asarray(data["train_features"][query_rows], dtype=np.float32)
                ).to(resolved_device)
                coordinates = model(features)
                delta = coordinates @ basis
                lambda_ = float(rngs[corpus_index].choice(train_lambdas))
                corrected = F.normalize(queries + lambda_ * delta, dim=1)
                positive_docs = _document_batch(
                    data["documents"], positive_indices, resolved_device
                )
                negative_docs = _document_batch(
                    data["documents"], negative_indices, resolved_device
                )
                positive_scores = torch.einsum(
                    "bd,bpd->bp", corrected, positive_docs
                )
                negative_scores = torch.einsum(
                    "bd,bnd->bn", corrected, negative_docs
                )
                rank_loss = multi_positive_hard_negative_loss(
                    positive_scores,
                    negative_scores,
                    torch.from_numpy(positive_indices >= 0).to(resolved_device),
                    torch.from_numpy(negative_indices >= 0).to(resolved_device),
                    temperature=temperature,
                )
                loss = rank_loss + float(coordinate_penalty) * torch.mean(
                    coordinates**2
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_losses.append(float(rank_loss.detach().cpu()))
        history.append(float(np.mean(epoch_losses)))
    return model, history


def predict_coordinates(model: LinearRankCorrection, features: np.ndarray) -> np.ndarray:
    device = next(model.parameters()).device
    with torch.no_grad():
        return (
            model(torch.from_numpy(np.asarray(features, dtype=np.float32)).to(device))
            .cpu()
            .numpy()
        )


__all__ = [
    "LinearRankCorrection",
    "train_linear_rank_model",
    "predict_coordinates",
]
