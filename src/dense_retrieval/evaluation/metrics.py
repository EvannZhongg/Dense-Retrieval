from __future__ import annotations

import math
from typing import Sequence


def _dcg(relevances: Sequence[int]) -> float:
    return sum((2 ** int(rel) - 1) / math.log2(rank + 2) for rank, rel in enumerate(relevances))


def compute_metrics(query_ids, qrels, ranked_document_ids, cutoffs=(1, 5, 10)) -> dict:
    if len(query_ids) != len(ranked_document_ids):
        raise ValueError("query_ids and rankings must have equal length")
    hits = {int(k): [] for k in cutoffs}
    recalls = {int(k): [] for k in cutoffs}
    reciprocal = []
    ndcgs = []
    for query_id, ranking in zip(query_ids, ranked_document_ids):
        relevant = {str(d): int(v) for d, v in qrels.get(str(query_id), {}).items() if int(v) > 0}
        if not relevant:
            raise ValueError(f"Query {query_id} has no positive relevance judgments")
        ranking = [str(d) for d in ranking]
        ranks = [i + 1 for i, doc_id in enumerate(ranking) if doc_id in relevant]
        reciprocal.append(1.0 / ranks[0] if ranks and ranks[0] <= 10 else 0.0)
        gains = [relevant.get(doc_id, 0) for doc_id in ranking[:10]]
        ideal = sorted(relevant.values(), reverse=True)[:10]
        ndcgs.append(_dcg(gains) / _dcg(ideal) if _dcg(ideal) else 0.0)
        for cutoff in cutoffs:
            found = len(set(ranking[:cutoff]) & set(relevant))
            hits[int(cutoff)].append(float(found > 0))
            recalls[int(cutoff)].append(found / len(relevant))
    result = {"num_queries": len(query_ids)}
    for cutoff in cutoffs:
        result[f"HitRate@{cutoff}"] = sum(hits[int(cutoff)]) / len(query_ids)
        result[f"Recall@{cutoff}"] = sum(recalls[int(cutoff)]) / len(query_ids)
    result["MRR@10"] = sum(reciprocal) / len(query_ids)
    result["NDCG@10"] = sum(ndcgs) / len(query_ids)
    return result
