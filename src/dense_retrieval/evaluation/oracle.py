from __future__ import annotations

import numpy as np
import pandas as pd

from ..retrieval.exact import exact_search


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
        positive_indices = [document_index[doc_id] for doc_id in relevant_ids]
        original_rank = min(
            int(np.where(baseline_indices[query_index] == index)[0][0]) + 1
            for index in positive_indices
        )
        similarities = (
            query_embeddings[query_index] @ document_embeddings[positive_indices].T
        )
        positive = document_embeddings[positive_indices[int(np.argmax(similarities))]]

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
        frame["failure_group"] = frame["original_rank"].map(
            lambda rank: "hit"
            if rank <= 10
            else "near_miss"
            if rank <= 100
            else "hard_miss"
        )
    return frame

