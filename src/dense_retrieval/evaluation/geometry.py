from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import pandas as pd


def _softmax_entropy(scores: np.ndarray, temperature: float) -> float:
    values = scores / max(float(temperature), 1e-8)
    values -= np.max(values)
    probabilities = np.exp(values)
    probabilities /= np.maximum(probabilities.sum(), 1e-12)
    return float(
        -(probabilities * np.log(np.maximum(probabilities, 1e-12))).sum()
    )


def compute_geometry_metrics(
    query_ids: Sequence[str],
    qrels: Dict[str, Dict[str, int]],
    query_embeddings: np.ndarray,
    document_embeddings: np.ndarray,
    document_ids: Sequence[str],
    ranked_scores: np.ndarray,
    temperature: float = 1.0,
) -> pd.DataFrame:
    document_index = {str(doc_id): index for index, doc_id in enumerate(document_ids)}
    rows = []

    for query_index, query_id in enumerate(query_ids):
        positive_indices = [
            document_index[str(doc_id)]
            for doc_id, relevance in qrels[str(query_id)].items()
            if int(relevance) > 0 and str(doc_id) in document_index
        ]
        scores = ranked_scores[query_index]
        top1 = float(scores[0])
        if positive_indices:
            positive_embeddings = document_embeddings[positive_indices]
            positive_scores = query_embeddings[query_index] @ positive_embeddings.T
            best_positive = float(np.max(positive_scores))
            mean_positive = float(np.mean(positive_scores))
            positive_norm = float(np.mean(np.linalg.norm(positive_embeddings, axis=1)))
        else:
            best_positive = None
            mean_positive = None
            positive_norm = None

        rows.append(
            {
                "query_id": str(query_id),
                "top1_score": top1,
                "top5_scores": scores[:5].tolist(),
                "top10_scores": scores[:10].tolist(),
                "best_relevant_score": best_positive,
                "top1_minus_relevant_score": (
                    top1 - best_positive if best_positive is not None else None
                ),
                "top1_top2_margin": top1 - float(scores[min(1, len(scores) - 1)]),
                "top1_top10_margin": top1 - float(scores[min(9, len(scores) - 1)]),
                "best_positive_similarity": best_positive,
                "mean_positive_similarity": mean_positive,
                "top5_mean": float(np.mean(scores[:5])),
                "top10_mean": float(np.mean(scores[:10])),
                "top20_mean": float(np.mean(scores[:20])),
                "score_std_top10": float(np.std(scores[:10])),
                "score_std_top20": float(np.std(scores[:20])),
                "entropy_top10": _softmax_entropy(scores[:10], temperature),
                "entropy_top20": _softmax_entropy(scores[:20], temperature),
                "entropy_top50": _softmax_entropy(scores[:50], temperature),
                "local_distance_5": float(np.mean(1 - scores[:5])),
                "local_distance_10": float(np.mean(1 - scores[:10])),
                "local_distance_20": float(np.mean(1 - scores[:20])),
                "query_embedding_norm": float(np.linalg.norm(query_embeddings[query_index])),
                "positive_embedding_norm": positive_norm,
            }
        )
    return pd.DataFrame(rows)
