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

