import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from dense_retrieval.analysis.local_geometry import (
    CorpusPrototypeGeometry,
    fit_reference_geometry,
    projected_cell_moments,
    prototype_counts,
    query_correction_features,
    query_local_geometry_features,
)


def test_prototype_counts_cover_all_documents():
    documents = np.asarray([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])
    prototypes = np.eye(2)

    counts = prototype_counts(documents, prototypes, batch_size=2)

    assert counts.tolist() == [2, 1]


def test_local_features_are_invariant_to_prototype_order():
    geometry = CorpusPrototypeGeometry(
        np.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]),
        np.asarray([4.0, 2.0, 1.0]),
    )
    permutation = np.asarray([2, 0, 1])
    permuted = CorpusPrototypeGeometry(
        geometry.prototypes[permutation], geometry.occupancy[permutation]
    )
    queries = np.asarray([[0.8, 0.2], [0.1, 0.9]])
    projection = np.eye(2)

    first = query_local_geometry_features(
        queries, geometry, projection, top_m=3, temperature=0.2
    )
    second = query_local_geometry_features(
        queries, permuted, projection, top_m=3, temperature=0.2
    )

    assert first.shape == (2, 10)
    assert np.allclose(first, second)


def test_local_features_change_with_corpus_geometry():
    queries = np.asarray([[1.0, 0.0]])
    first = CorpusPrototypeGeometry(np.asarray([[1.0, 0.0], [0.0, 1.0]]), [1, 1])
    second = CorpusPrototypeGeometry(np.asarray([[0.0, 1.0], [-1.0, 0.0]]), [1, 1])

    first_features = query_local_geometry_features(
        queries, first, np.eye(2), top_m=2
    )
    second_features = query_local_geometry_features(
        queries, second, np.eye(2), top_m=2
    )

    assert not np.allclose(first_features, second_features)


def test_reference_geometry_and_feature_controls_are_well_formed():
    documents = {
        "a": np.asarray([[1.0, 0.0], [0.8, 0.2]]),
        "b": np.asarray([[0.0, 1.0], [0.2, 0.8]]),
    }
    reference = fit_reference_geometry(
        documents,
        n_prototypes=2,
        max_fit_documents=4,
        max_iter=5,
        random_state=3,
        assignment_batch_size=2,
    )
    local = CorpusPrototypeGeometry(documents["a"], [1, 1])
    queries = np.asarray([[1.0, 0.0]])

    q_only = query_correction_features(
        queries, "q_only", local, reference, np.eye(2), top_m=2, temperature=0.1
    )
    corpus = query_correction_features(
        queries,
        "corpus_geometry",
        local,
        reference,
        np.eye(2),
        top_m=2,
        temperature=0.1,
    )

    assert q_only.shape == (1, 2)
    assert corpus.shape == (1, 12)
    assert np.isclose(reference.occupancy.sum(), 1.0)


def test_projected_cell_moments_extend_local_features():
    documents = np.asarray(
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]], dtype=float
    )
    geometry = CorpusPrototypeGeometry(np.eye(2), [2, 2])
    moments = projected_cell_moments(
        documents,
        geometry,
        np.eye(2),
        max_documents=4,
        batch_size=2,
        random_state=0,
    )
    features = query_local_geometry_features(
        np.asarray([[1.0, 0.0]]),
        geometry,
        np.eye(2),
        top_m=2,
        cell_moments=moments,
    )

    assert moments.mean.shape == (2, 2)
    assert moments.variance.shape == (2, 2)
    assert np.all(moments.variance >= 0)
    assert features.shape == (1, 14)
