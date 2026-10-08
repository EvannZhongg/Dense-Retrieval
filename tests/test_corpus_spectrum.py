import numpy as np

from dense_retrieval.analysis.corpus_spectrum import (
    document_spectrum,
    fit_shared_document_basis,
    query_spectral_interactions,
)


def test_document_spectrum_is_document_only_and_query_features_are_aligned():
    documents_a = np.array([[1.0, 0.0], [2.0, 0.0]])
    documents_b = np.array([[0.0, 1.0], [0.0, 2.0]])
    mean, components = fit_shared_document_basis([documents_a, documents_b], rank=2)
    stats = document_spectrum(documents_a, mean, components)
    features = query_spectral_interactions(
        np.array([[1.0, 0.0], [0.0, 1.0]]), stats, mean, components
    )

    assert components.shape == (2, 2)
    assert stats["variance"].shape == (2,)
    assert features.shape == (2, 6)
    assert np.all(np.isfinite(features))


def test_document_spectrum_rejects_mismatched_basis():
    mean, components = fit_shared_document_basis([np.eye(3)], rank=2)
    try:
        document_spectrum(np.eye(2), mean, components)
    except ValueError as error:
        assert "incompatible" in str(error)
    else:
        raise AssertionError("expected a shape validation error")
