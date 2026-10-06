import numpy as np
import pytest

from dense_retrieval.analysis.prototype_correction import (
    PrototypeCorrectionField,
    fit_prototype_values,
    fit_spherical_kmeans,
    predict_prototype_values,
    prototype_weights,
)


def test_spherical_kmeans_returns_unit_centers_and_is_deterministic():
    rng = np.random.default_rng(4)
    samples = np.vstack(
        [rng.normal(loc=center, scale=0.03, size=(30, 3)) for center in ((1, 0, 0), (0, 1, 0))]
    )
    first = fit_spherical_kmeans(samples, n_clusters=2, n_init=3, random_state=11)
    second = fit_spherical_kmeans(samples, n_clusters=2, n_init=3, random_state=11)
    assert np.allclose(np.linalg.norm(first, axis=1), 1.0)
    assert np.allclose(first, second)
    assert np.max(np.abs(first @ first.T - np.eye(2))) < 0.1


def test_top_m_weights_are_sparse_and_normalized_on_support():
    queries = np.array([[1.0, 0.0], [0.0, 1.0]])
    prototypes = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    weights = prototype_weights(queries, prototypes, temperature=0.1, top_m=2)
    assert np.allclose(weights.sum(axis=1), 1.0)
    assert np.all(np.count_nonzero(weights, axis=1) == 2)
    assert np.argmax(weights[0]) == 0
    assert np.argmax(weights[1]) == 1


def test_prototype_values_reconstruct_linear_targets():
    weights = np.eye(3)
    values = np.array([[1.0, 2.0], [-1.0, 0.5], [0.2, 3.0]])
    fitted = fit_prototype_values(weights, values)
    assert np.allclose(fitted, values)
    assert np.allclose(predict_prototype_values(weights, fitted), values)


def test_field_predicts_and_applies_normalized_query_correction():
    documents = np.eye(3)
    queries = np.eye(3)
    targets = np.array([[0.5, 0.0], [0.0, 0.25], [0.1, 0.2]])
    field = PrototypeCorrectionField.fit(
        documents,
        queries,
        targets,
        n_prototypes=3,
        top_m=1,
        temperature=0.1,
        ridge=0.0,
        random_state=0,
        mean=np.zeros(3),
        components=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
    )
    assert np.allclose(field.predict_coordinates(queries), targets, atol=1e-6)
    corrected = field.apply(queries, lambda_=0.5)
    assert np.allclose(np.linalg.norm(corrected, axis=1), 1.0)


def test_invalid_top_m_and_zero_rows_are_rejected():
    with pytest.raises(ValueError):
        prototype_weights(np.zeros((1, 2)), np.eye(2), top_m=1)
    with pytest.raises(ValueError):
        prototype_weights(np.ones((1, 2)), np.eye(2), top_m=3)
