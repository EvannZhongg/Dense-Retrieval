import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from dense_retrieval.analysis.local_geometry import (
    CorpusPrototypeGeometry,
    prototype_counts,
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
