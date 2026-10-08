"""Diagnostics for non-identifiability of privileged ranking optima."""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from ..evaluation.quality import compute_quality_metrics
from ..retrieval.exact import exact_search


_EPS = 1e-12


def subsample_rank_records(
    records: dict[str, np.ndarray],
    negative_count: int,
    *,
    random_state: int | None = None,
) -> dict[str, np.ndarray]:
    """Select a fixed number of negatives per query from a larger hard pool.

    With ``random_state=None`` this selects the hardest prefix. With a seed it
    samples independently per query without replacement, preserving a clean
    distinction between negative-count and negative-sample perturbations.
    """
    if negative_count < 1:
        raise ValueError("negative_count must be positive")
    negatives = np.asarray(records["negative_indices"], dtype=np.int64)
    if negative_count > negatives.shape[1]:
        raise ValueError("negative_count exceeds the available hard-negative pool")
    selected = np.full((len(negatives), negative_count), -1, dtype=np.int64)
    rng = None if random_state is None else np.random.default_rng(random_state)
    for row, values in enumerate(negatives):
        available = values[values >= 0]
        count = min(negative_count, len(available))
        if rng is None:
            chosen = available[:count]
        else:
            chosen = rng.choice(available, size=count, replace=False)
        selected[row, :count] = chosen
    return {
        "query_rows": np.asarray(records["query_rows"], dtype=np.int64).copy(),
        "positive_indices": np.asarray(
            records["positive_indices"], dtype=np.int64
        ).copy(),
        "negative_indices": selected,
    }


def per_query_retrieval_metrics(
    dataset,
    corrected_queries: np.ndarray,
    documents: np.ndarray,
    rank_records: dict[str, np.ndarray],
    *,
    top_k: int = 10,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Return per-query ranking metrics, positive margin, and Top-K indices."""
    rows = np.asarray(rank_records["query_rows"], dtype=np.int64)
    queries = np.asarray(corrected_queries, dtype=np.float32)
    if len(queries) != len(rows):
        raise ValueError("corrected_queries must align with rank_records query_rows")
    k = min(int(top_k), len(documents))
    rankings, _ = exact_search(queries, documents, k)
    document_ids = np.asarray(list(dataset.corpus), dtype=object)
    ranked_ids = document_ids[rankings].tolist()
    query_ids = [str(dataset.queries[row].query_id) for row in rows]
    _, quality = compute_quality_metrics(
        query_ids,
        dataset.qrels,
        ranked_ids,
        cutoffs=(top_k,),
        available_document_ids=set(dataset.corpus),
    )

    positive_indices = np.asarray(rank_records["positive_indices"], dtype=np.int64)
    positive_margins = []
    search_k = min(len(documents), positive_indices.shape[1] + 1)
    candidates, candidate_scores = exact_search(queries, documents, search_k)
    for row in range(len(queries)):
        positives = positive_indices[row][positive_indices[row] >= 0]
        positive_score = float(np.max(documents[positives] @ queries[row]))
        positive_set = set(positives.tolist())
        negative_score = next(
            float(score)
            for index, score in zip(candidates[row], candidate_scores[row])
            if int(index) not in positive_set
        )
        positive_margins.append(positive_score - negative_score)
    result = quality[["query_id", "ndcg_at_10", "reciprocal_rank_at_10"]].copy()
    result["positive_margin"] = positive_margins
    return result, rankings


def per_query_ranking_loss(
    corrected_queries: np.ndarray,
    documents: np.ndarray,
    rank_records: dict[str, np.ndarray],
    *,
    temperature: float = 0.05,
) -> np.ndarray:
    """Evaluate every solution against one common ranking objective."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    queries = np.asarray(corrected_queries, dtype=np.float64)
    positives = np.asarray(rank_records["positive_indices"], dtype=np.int64)
    negatives = np.asarray(rank_records["negative_indices"], dtype=np.int64)
    if len(queries) != len(positives):
        raise ValueError("corrected_queries must align with rank_records")
    positive_mask = positives >= 0
    negative_mask = negatives >= 0
    positive_scores = np.einsum(
        "bd,bpd->bp", queries, documents[np.maximum(positives, 0)]
    ) / float(temperature)
    negative_scores = np.einsum(
        "bd,bnd->bn", queries, documents[np.maximum(negatives, 0)]
    ) / float(temperature)
    positive_scores = np.where(positive_mask, positive_scores, -np.inf)
    negative_scores = np.where(negative_mask, negative_scores, -np.inf)

    def logsumexp(values: np.ndarray) -> np.ndarray:
        maximum = np.max(values, axis=1, keepdims=True)
        return maximum[:, 0] + np.log(
            np.sum(np.exp(values - maximum), axis=1)
        )

    numerator = logsumexp(positive_scores)
    denominator = logsumexp(
        np.concatenate([positive_scores, negative_scores], axis=1)
    )
    return -(numerator - denominator)


def _row_cosines(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    numerator = np.sum(left * right, axis=1)
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    return np.divide(
        numerator,
        denominator,
        out=np.full(len(left), np.nan, dtype=np.float64),
        where=denominator > _EPS,
    )


def pairwise_solution_stability(solutions: list[dict]) -> pd.DataFrame:
    """Compare all solution pairs belonging to the same perturbation group."""
    output = []
    groups = sorted({str(item["comparison_group"]) for item in solutions})
    for group in groups:
        members = [
            item for item in solutions if str(item["comparison_group"]) == group
        ]
        for left, right in combinations(members, 2):
            if not np.array_equal(left["query_rows"], right["query_rows"]):
                raise ValueError("all solutions must use identical query rows")
            coordinate_cosine = _row_cosines(
                left["coordinates"], right["coordinates"]
            )
            displacement_cosine = _row_cosines(left["deltas"], right["deltas"])
            left_metrics = left["metrics"].set_index("query_id")
            right_metrics = right["metrics"].set_index("query_id")
            for row, query_id in enumerate(left["query_ids"]):
                left_top = set(map(int, left["rankings"][row]))
                right_top = set(map(int, right["rankings"][row]))
                overlap = len(left_top & right_top) / max(len(left_top), 1)
                left_values = left_metrics.loc[query_id]
                right_values = right_metrics.loc[query_id]
                output.append(
                    {
                        "comparison_group": group,
                        "left_run": left["run_id"],
                        "right_run": right["run_id"],
                        "query_id": query_id,
                        "coordinate_cosine": coordinate_cosine[row],
                        "displacement_cosine": displacement_cosine[row],
                        "top_k_overlap": overlap,
                        "left_ndcg_at_10": left_values["ndcg_at_10"],
                        "right_ndcg_at_10": right_values["ndcg_at_10"],
                        "abs_ndcg_difference": abs(
                            left_values["ndcg_at_10"]
                            - right_values["ndcg_at_10"]
                        ),
                        "left_mrr_at_10": left_values["reciprocal_rank_at_10"],
                        "right_mrr_at_10": right_values["reciprocal_rank_at_10"],
                        "abs_mrr_difference": abs(
                            left_values["reciprocal_rank_at_10"]
                            - right_values["reciprocal_rank_at_10"]
                        ),
                        "left_positive_margin": left_values["positive_margin"],
                        "right_positive_margin": right_values["positive_margin"],
                        "abs_positive_margin_difference": abs(
                            left_values["positive_margin"]
                            - right_values["positive_margin"]
                        ),
                        "left_reference_rank_loss": left_values[
                            "reference_rank_loss"
                        ],
                        "right_reference_rank_loss": right_values[
                            "reference_rank_loss"
                        ],
                        "abs_reference_rank_loss_difference": abs(
                            left_values["reference_rank_loss"]
                            - right_values["reference_rank_loss"]
                        ),
                    }
                )
    return pd.DataFrame(output)


def summarize_pairwise_stability(pairwise: pd.DataFrame) -> pd.DataFrame:
    """Aggregate directional and retrieval stability without hiding tails."""
    keys = ["heldout_corpus", "model", "comparison_group"]
    rows = []
    for values, group in pairwise.groupby(keys, sort=True):
        directionally_distinct = group["coordinate_cosine"] < 0.2
        metric_equivalent = (
            (group["abs_ndcg_difference"] <= 0.01)
            & (group["abs_mrr_difference"] <= 0.01)
        )
        ranking_equivalent = metric_equivalent & (group["top_k_overlap"] >= 0.8)
        rows.append(
            {
                **dict(zip(keys, values)),
                "pair_query_count": len(group),
                "mean_coordinate_cosine": group["coordinate_cosine"].mean(),
                "p10_coordinate_cosine": group["coordinate_cosine"].quantile(0.1),
                "mean_displacement_cosine": group["displacement_cosine"].mean(),
                "mean_top_k_overlap": group["top_k_overlap"].mean(),
                "mean_abs_ndcg_difference": group["abs_ndcg_difference"].mean(),
                "mean_abs_mrr_difference": group["abs_mrr_difference"].mean(),
                "mean_abs_positive_margin_difference": group[
                    "abs_positive_margin_difference"
                ].mean(),
                "mean_abs_reference_rank_loss_difference": group[
                    "abs_reference_rank_loss_difference"
                ].mean(),
                "directionally_distinct_fraction": float(
                    directionally_distinct.mean()
                ),
                "directionally_distinct_but_metric_equivalent_fraction": float(
                    (directionally_distinct & metric_equivalent).mean()
                ),
                "directionally_distinct_but_ranking_equivalent_fraction": float(
                    (directionally_distinct & ranking_equivalent).mean()
                ),
            }
        )
    return pd.DataFrame(rows)


__all__ = [
    "pairwise_solution_stability",
    "per_query_ranking_loss",
    "per_query_retrieval_metrics",
    "subsample_rank_records",
    "summarize_pairwise_stability",
]
