import numpy as np

from dense_retrieval.datasets.base import DatasetSample, Document, RetrievalDataset
from dense_retrieval.evaluation import evaluate_baseline


def test_exact_frozen_baseline_metrics():
    dataset = RetrievalDataset("toy", [DatasetSample("q1", "q", ["d1"])], {"d1": Document("d1", "", ""), "d2": Document("d2", "", "")}, {"q1": {"d1": 1}})
    metrics = evaluate_baseline(dataset, np.array([[1.0, 0.0]], dtype=np.float32), np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32), ["d1", "d2"], 2)
    assert metrics["HitRate@5"] == 1.0
    assert metrics["MRR@10"] == 1.0
