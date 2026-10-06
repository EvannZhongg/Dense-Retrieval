from __future__ import annotations

import math
from typing import Dict, Sequence

import numpy as np
import pandas as pd


DEFAULT_CUTOFFS = (5, 10, 20, 50, 100)


def _dcg(relevances: Sequence[int]) -> float:
    return sum(
        (2**relevance - 1) / math.log2(rank + 2)
        for rank, relevance in enumerate(relevances)
    )


def compute_quality_metrics(
    query_ids: Sequence[str],
    qrels: Dict[str, Dict[str, int]],
    ranked_document_ids: Sequence[Sequence[str]],
    cutoffs: Sequence[int] = DEFAULT_CUTOFFS,
    available_document_ids: set[str] | None = None,
) -> tuple[dict, pd.DataFrame]:
    """Compute macro retrieval metrics and auditable per-query values.

    HitRate@K is the fraction of queries with at least one relevant document in
    top K. Recall@K is the macro mean of retrieved_relevant / relevant.
    """
    if len(query_ids) != len(ranked_document_ids):
        raise ValueError("query_ids and ranked_document_ids must have equal length")

    rows = []
    reciprocal_ranks = []
    ndcg_values = []
    cutoffs = tuple(sorted(set(int(cutoff) for cutoff in cutoffs)))

    for query_id, ranking in zip(query_ids, ranked_document_ids):
        query_id = str(query_id)
        relevant = {
            str(doc_id): int(relevance)
            for doc_id, relevance in qrels.get(query_id, {}).items()
            if int(relevance) > 0
        }
        if not relevant:
            raise ValueError(f"Query {query_id} has no positive relevance judgments")

        ranking = [str(doc_id) for doc_id in ranking]
        relevant_ranks = [
            rank for rank, doc_id in enumerate(ranking, start=1) if doc_id in relevant
        ]
        best_rank = min(relevant_ranks, default=None)
        available_ids = available_document_ids if available_document_ids is not None else set(ranking)
        row = {
            "query_id": query_id,
            "relevant_doc_ids": list(relevant),
            "relevant_count": len(relevant),
            "available_relevant_count": sum(doc_id in available_ids for doc_id in relevant),
            "missing_relevant_count": sum(doc_id not in available_ids for doc_id in relevant),
            "best_relevant_rank": best_rank,
        }

        relevant_ids = set(relevant)
        for cutoff in cutoffs:
            retrieved_count = len(relevant_ids.intersection(ranking[:cutoff]))
            row[f"retrieved_relevant_at_{cutoff}"] = retrieved_count
            row[f"hit_at_{cutoff}"] = retrieved_count > 0
            row[f"recall_at_{cutoff}"] = retrieved_count / len(relevant)

        reciprocal_ranks.append(
            1.0 / best_rank if best_rank is not None and best_rank <= 10 else 0.0
        )
        gains = [relevant.get(doc_id, 0) for doc_id in ranking[:10]]
        ideal = sorted(relevant.values(), reverse=True)[:10]
        ideal_dcg = _dcg(ideal)
        ndcg_values.append(_dcg(gains) / ideal_dcg if ideal_dcg else 0.0)
        rows.append(row)

    frame = pd.DataFrame(rows)
    metrics = {}
    for cutoff in cutoffs:
        metrics[f"HitRate@{cutoff}"] = float(frame[f"hit_at_{cutoff}"].mean())
        metrics[f"Recall@{cutoff}"] = float(frame[f"recall_at_{cutoff}"].mean())
    metrics.update(
        {
            "MRR@10": float(np.mean(reciprocal_ranks)),
            "NDCG@10": float(np.mean(ndcg_values)),
            "num_queries": len(frame),
        }
    )
    return metrics, frame
