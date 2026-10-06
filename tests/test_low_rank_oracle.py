import numpy as np

from dense_retrieval.analysis.query_correction import (
    apply_correction,
    fit_pca_subspace,
    gain_retention,
    project_deltas,
)


def test_projection_uses_only_fitted_mean_and_basis():
    train = np.array(
        [[0.0, -2.0, 5.0], [0.0, -1.0, 5.0], [0.0, 1.0, 5.0], [0.0, 2.0, 5.0]]
    )
    mean, components = fit_pca_subspace(train, max_rank=1)
    test = np.array([[9.0, 3.0, -4.0]])
    projected = project_deltas(test, mean, components, rank=1)

    assert np.allclose(mean, [0.0, 0.0, 5.0])
    assert np.allclose(projected, [[0.0, 3.0, 5.0]])


def test_gain_retention():
    assert np.isclose(gain_retention(0.92, 0.681, 1.0), 0.7492163)
    assert np.isnan(gain_retention(0.5, 0.5, 0.5))


def test_apply_correction_leaves_queries_without_oracle_delta_unchanged():
    queries = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    rows = np.array([0])
    deltas = np.array([[-1.0, 1.0]])
    corrected = apply_correction(queries, rows, deltas, 0.5)

    assert np.allclose(corrected[0], [2**-0.5, 2**-0.5])
    assert np.allclose(corrected[1], queries[1])
