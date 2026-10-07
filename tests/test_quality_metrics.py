import sys
sys.path.insert(0, "src")

from dense_retrieval.evaluation.quality import compute_quality_metrics


def test_hit_rate_and_recall_are_distinct_for_multi_positive_queries():
    metrics, per_query = compute_quality_metrics(
        query_ids=["q1", "q2"],
        qrels={"q1": {"a": 1, "b": 1}, "q2": {"c": 1}},
        ranked_document_ids=[["a", "x"], ["c", "y"]],
        cutoffs=[1],
    )
    assert metrics["HitRate@1"] == 1.0
    assert metrics["Recall@1"] == 0.75
    assert per_query["retrieved_relevant_at_1"].tolist() == [1, 1]
    assert per_query["recall_at_1"].tolist() == [0.5, 1.0]


def test_missing_relevant_document_is_auditable_and_counts_as_miss():
    metrics, per_query = compute_quality_metrics(
        query_ids=["q"],
        qrels={"q": {"missing": 1}},
        ranked_document_ids=[["available"]],
        cutoffs=[1],
        available_document_ids={"available"},
    )
    assert metrics["HitRate@1"] == 0.0
    assert metrics["Recall@1"] == 0.0
    assert per_query.loc[0, "available_relevant_count"] == 0
    assert per_query.loc[0, "missing_relevant_count"] == 1


def test_per_query_frame_exposes_metrics_used_by_benefit_gating():
    metrics, per_query = compute_quality_metrics(
        query_ids=["q"],
        qrels={"q": {"relevant": 2}},
        ranked_document_ids=[["other", "relevant"]],
        cutoffs=[1, 10],
    )

    assert per_query.loc[0, "reciprocal_rank_at_10"] == 0.5
    assert per_query.loc[0, "ndcg_at_10"] == metrics["NDCG@10"]
