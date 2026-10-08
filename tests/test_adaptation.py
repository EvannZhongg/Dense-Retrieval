import numpy as np

from dense_retrieval.adaptation import LinearQueryCalibrator, fit_corpus_alignment, fit_gold_supervised


def test_corpus_alignment_maps_query_protocol_vectors_to_frozen_documents():
    source = np.eye(3, dtype=np.float32)
    target = np.roll(source, 1, axis=1)
    calibrator = fit_corpus_alignment(source, target, alpha=1e-6)
    transformed = calibrator.transform(source)
    assert np.argmax(transformed, axis=1).tolist() == [1, 2, 0]


def test_gold_fit_uses_only_declared_training_queries():
    queries = np.eye(3, dtype=np.float32)
    documents = np.eye(3, dtype=np.float32)
    calibrator = fit_gold_supervised(
        queries,
        documents,
        ["q1", "q2", "q3"],
        {"q1": {"d1": 1}, "q2": {"d2": 1}, "q3": {"d3": 1}},
        ["d1", "d2", "d3"],
        train_query_ids=["q1", "q2"],
        alpha=1e-3,
    )
    transformed = calibrator.transform(queries)
    assert np.argmax(transformed[:2], axis=1).tolist() == [0, 1]


def test_calibrator_round_trip(tmp_path):
    calibrator = LinearQueryCalibrator(np.eye(2, dtype=np.float32))
    path = calibrator.save(tmp_path / "calibrator.npz", metadata={"dataset": "toy"})
    loaded = LinearQueryCalibrator.load(path)
    assert loaded.method == "ridge"
    assert np.allclose(loaded.transform([[1.0, 0.0]]), [[1.0, 0.0]])
