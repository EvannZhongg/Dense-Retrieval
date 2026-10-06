from __future__ import annotations

import numpy as np
import pandas as pd

from ..retrieval.exact import exact_search


def closest_positive_index(
    query_id,
    qrels,
    query_embedding,
    document_embeddings,
    document_index,
):
    """Return the available positive with maximum query cosine similarity."""
    positive_indices = [
        document_index[str(doc_id)]
        for doc_id, relevance in qrels[str(query_id)].items()
        if int(relevance) > 0 and str(doc_id) in document_index
    ]
    if not positive_indices:
        return None
    similarities = query_embedding @ document_embeddings[positive_indices].T
    return positive_indices[int(np.argmax(similarities))]


def run_oracle_correction(
    query_ids,
    qrels,
    query_embeddings,
    document_embeddings,
    document_ids,
    lambdas=(0, 0.01, 0.02, 0.05, 0.1, 0.2),
    search_k=100,
):
    document_ids = list(map(str, document_ids))
    document_index = {doc_id: index for index, doc_id in enumerate(document_ids)}
    output = []
    baseline_indices, _ = exact_search(
        query_embeddings, document_embeddings, len(document_ids)
    )

    for query_index, query_id in enumerate(query_ids):
        relevant_ids = {
            str(doc_id)
            for doc_id, relevance in qrels[str(query_id)].items()
            if int(relevance) > 0
        }
        available_relevant_ids = relevant_ids.intersection(document_index)
        positive_index = closest_positive_index(
            query_id,
            qrels,
            query_embeddings[query_index],
            document_embeddings,
            document_index,
        )
        if positive_index is None:
            failure_group = "hard_miss"
            for lambda_ in lambdas:
                row = {
                    "query_id": str(query_id),
                    "lambda": float(lambda_),
                    "relevant_count": len(relevant_ids),
                    "available_relevant_count": 0,
                    "missing_relevant_count": len(relevant_ids),
                    "correction_available": False,
                    "original_rank": None,
                    "corrected_rank": None,
                    "rank_improvement": None,
                    "failure_group": failure_group,
                }
                for cutoff in (5, 10, 20, 100):
                    row[f"retrieved_relevant_at_{cutoff}"] = 0
                    row[f"entered_top{cutoff}"] = False
                    row[f"recall_at_{cutoff}"] = 0.0
                output.append(row)
            continue
        original_rank = min(
            int(np.where(baseline_indices[query_index] == index)[0][0]) + 1
            for index in (document_index[doc_id] for doc_id in available_relevant_ids)
        )
        positive = document_embeddings[positive_index]

        for lambda_ in lambdas:
            corrected = query_embeddings[query_index] * (1 - lambda_) + positive * lambda_
            corrected /= max(np.linalg.norm(corrected), 1e-12)
            indices, _ = exact_search(
                corrected[None, :],
                document_embeddings,
                min(search_k, len(document_ids)),
            )
            ranking = [document_ids[index] for index in indices[0]]
            relevant_ranks = [
                rank
                for rank, doc_id in enumerate(ranking, start=1)
                if doc_id in relevant_ids
            ]
            corrected_rank = min(relevant_ranks, default=None)
            row = {
                "query_id": str(query_id),
                "lambda": float(lambda_),
                "relevant_count": len(relevant_ids),
                "available_relevant_count": len(available_relevant_ids),
                "missing_relevant_count": len(relevant_ids - available_relevant_ids),
                "correction_available": True,
                "original_rank": original_rank,
                "corrected_rank": corrected_rank,
                "rank_improvement": (
                    original_rank - corrected_rank
                    if corrected_rank is not None
                    else None
                ),
            }
            for cutoff in (5, 10, 20, 100):
                count = sum(doc_id in relevant_ids for doc_id in ranking[:cutoff])
                row[f"retrieved_relevant_at_{cutoff}"] = count
                row[f"entered_top{cutoff}"] = count > 0
                row[f"recall_at_{cutoff}"] = count / len(relevant_ids)
            output.append(row)

    frame = pd.DataFrame(output)
    if not frame.empty:
        def classify(rank):
            if pd.isna(rank):
                return "hard_miss"
            if rank <= 10:
                return "hit"
            if rank <= 100:
                return "near_miss"
            return "hard_miss"

        frame["failure_group"] = frame["original_rank"].map(classify)
    return frame
