import numpy as np

from dense_retrieval.analysis.action_space import optimize_ranking_oracle_solution
from dense_retrieval.analysis.oracle_stability import (
    pairwise_solution_stability,
    per_query_ranking_loss,
    subsample_rank_records,
)


def test_subsample_rank_records_separates_prefix_and_seeded_sample():
    records = {
        "query_rows": np.array([2]),
        "positive_indices": np.array([[0]]),
        "negative_indices": np.array([[1, 2, 3, 4, 5]]),
    }

    prefix = subsample_rank_records(records, 3)
    sample = subsample_rank_records(records, 3, random_state=7)

    assert prefix["negative_indices"].tolist() == [[1, 2, 3]]
    assert len(set(sample["negative_indices"][0])) == 3
    assert set(sample["negative_indices"][0]).issubset({1, 2, 3, 4, 5})


def test_oracle_solution_exposes_coordinates_and_respects_initial_shape():
    queries = np.array([[1.0, 0.0]], dtype=np.float32)
    documents = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    records = {
        "query_rows": np.array([0]),
        "positive_indices": np.array([[0]]),
        "negative_indices": np.array([[1]]),
    }
    basis = np.array([[0.0, 1.0]], dtype=np.float32)

    solution = optimize_ranking_oracle_solution(
        queries,
        documents,
        records,
        basis,
        steps=20,
        learning_rate=0.1,
        penalty=0.0,
        temperature=1.0,
        device="cpu",
        initial_coordinates=np.array([[0.2]], dtype=np.float32),
        optimizer_seed=4,
    )

    assert solution.coordinates.shape == (1, 1)
    assert np.allclose(solution.deltas, solution.coordinates @ basis)
    assert solution.rank_loss < np.log(2.0)


def test_pairwise_stability_can_detect_equivalent_retrieval_with_orthogonal_coordinates():
    import pandas as pd

    metrics = pd.DataFrame(
        {
            "query_id": ["q1"],
            "ndcg_at_10": [1.0],
            "reciprocal_rank_at_10": [1.0],
            "positive_margin": [0.2],
            "reference_rank_loss": [0.1],
        }
    )
    common = {
        "comparison_group": "initialization",
        "query_rows": np.array([0]),
        "query_ids": ["q1"],
        "metrics": metrics,
        "rankings": np.array([[0, 1, 2]]),
    }
    pairwise = pairwise_solution_stability(
        [
            {
                **common,
                "run_id": "a",
                "coordinates": np.array([[1.0, 0.0]]),
                "deltas": np.array([[1.0, 0.0]]),
            },
            {
                **common,
                "run_id": "b",
                "coordinates": np.array([[0.0, 1.0]]),
                "deltas": np.array([[0.0, 1.0]]),
            },
        ]
    )

    assert np.isclose(pairwise.loc[0, "coordinate_cosine"], 0.0)
    assert np.isclose(pairwise.loc[0, "top_k_overlap"], 1.0)
    assert np.isclose(pairwise.loc[0, "abs_ndcg_difference"], 0.0)


def test_per_query_ranking_loss_uses_a_common_reference_boundary():
    queries = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    documents = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    records = {
        "query_rows": np.array([0, 1]),
        "positive_indices": np.array([[0], [0]]),
        "negative_indices": np.array([[1], [1]]),
    }

    losses = per_query_ranking_loss(
        queries, documents, records, temperature=1.0
    )

    assert losses[0] < losses[1]
    assert np.all(np.isfinite(losses))
