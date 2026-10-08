import numpy as np

from dense_retrieval.analysis.action_space import (
    fit_svd_subspace,
    optimize_ranking_oracle,
    project_to_subspace,
    projection_statistics,
    random_subspace,
    tangent_projection,
)


def test_tangent_projection_removes_query_radial_component():
    queries = np.array([[1.0, 0.0], [0.0, 2.0]])
    vectors = np.array([[3.0, 4.0], [5.0, 6.0]])

    projected = tangent_projection(vectors, queries)

    unit_queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)
    assert np.allclose(np.sum(projected * unit_queries, axis=1), 0.0)
    assert np.allclose(projected, [[0.0, 4.0], [5.0, 0.0]])


def test_svd_subspace_prioritizes_sample_energy():
    samples = np.array([[4.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    components = fit_svd_subspace(samples, max_rank=2)
    projected = project_to_subspace(samples, components, rank=1)
    stats = projection_statistics(samples, projected)

    assert np.allclose(np.abs(components[0]), [1.0, 0.0, 0.0])
    assert np.isclose(stats["energy_retention"], 20.0 / 21.0)


def test_random_subspace_is_nested_orthonormal_and_reproducible():
    first = random_subspace(8, 4, random_state=7)
    second = random_subspace(8, 4, random_state=7)

    assert np.allclose(first, second)
    assert np.allclose(first @ first.T, np.eye(4))
    vector = np.arange(8, dtype=float)[None, :]
    rank_two = project_to_subspace(vector, first, rank=2)
    rank_four = project_to_subspace(vector, first, rank=4)
    assert np.linalg.norm(rank_four) >= np.linalg.norm(rank_two)


def test_ranking_oracle_improves_a_pairwise_boundary_inside_subspace():
    queries = np.array([[1.0, 0.0]], dtype=np.float32)
    documents = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    records = {
        "query_rows": np.array([0]),
        "positive_indices": np.array([[0]]),
        "negative_indices": np.array([[1]]),
    }

    rows, deltas, final_loss = optimize_ranking_oracle(
        queries,
        documents,
        records,
        components=np.array([[0.0, 1.0]], dtype=np.float32),
        steps=80,
        learning_rate=0.1,
        penalty=0.0,
        temperature=1.0,
        device="cpu",
    )

    assert rows.tolist() == [0]
    assert deltas[0, 1] > 0
    assert final_loss < np.log(2.0)
