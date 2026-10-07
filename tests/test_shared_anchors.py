import numpy as np

from dense_retrieval.analysis.shared_anchors import (
    SharedAnchorCodebook,
    compute_corpus_anchor_occupancy,
    fit_global_anchor_codebook,
)


def test_global_codebook_is_fitted_from_mapping_as_one_document_pool():
    first = {"b": np.eye(3), "a": np.eye(3) * 2.0}
    second = {"a": first["a"], "b": first["b"]}
    assert np.allclose(
        fit_global_anchor_codebook(first, n_anchors=3, random_state=4),
        fit_global_anchor_codebook(second, n_anchors=3, random_state=4),
    )


def test_query_fields_keep_query_part_fixed_and_change_with_occupancy():
    rng = np.random.default_rng(7)
    documents = {
        "small": rng.normal(size=(20, 4)),
        "large": rng.normal(size=(35, 4)),
    }
    codebook = SharedAnchorCodebook.fit(
        documents, n_anchors=5, top_m=3, random_state=9
    )
    small = codebook.occupancy(documents["small"])
    large = codebook.occupancy(documents["large"])
    queries = rng.normal(size=(6, 4))
    small_features = codebook.query_features(queries, small)
    large_features = codebook.query_features(queries, large)
    assert np.array_equal(small_features.indices, large_features.indices)
    assert np.allclose(small_features.values[..., :-1], large_features.values[..., :-1])
    assert not np.allclose(small_features.values[..., -1], large_features.values[..., -1])
    assert small.counts.sum() == len(documents["small"])
    assert large.counts.sum() == len(documents["large"])


def test_local_shape_statistics_are_aligned_to_anchor_cells():
    anchors = np.eye(3)
    documents = np.array(
        [[1.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.0, 1.0, 0.0]], dtype=float
    )
    stats = compute_corpus_anchor_occupancy(documents, anchors, projection=np.eye(3)[:2])
    assert stats.local_covariance.shape == (3, 2)
    assert np.isclose(stats.occupancy.sum(), 1.0)
    assert stats.counts.tolist() == [2, 1, 0]
