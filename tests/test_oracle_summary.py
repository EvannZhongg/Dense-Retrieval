import numpy as np

from dense_retrieval.analysis.query_correction import apply_oracle_correction


def test_apply_oracle_correction_moves_only_available_queries_and_normalizes():
    queries = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    positives = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    available = np.array([True, False])

    corrected = apply_oracle_correction(
        queries, positives, available, lambda_=0.5
    )

    assert np.allclose(corrected[0], [2**-0.5, 2**-0.5])
    assert np.allclose(corrected[1], queries[1])
    assert np.allclose(np.linalg.norm(corrected, axis=1), 1.0)
