import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from dense_retrieval.analysis.candidate_actions import (
    action_query_features,
    corrected_query_grid,
    fit_candidate_actions,
    reshape_utilities,
)


def test_candidate_actions_are_balanced_reproducible_and_include_abstention():
    first = np.tile([[1.0, 0.0]], (20, 1))
    second = np.tile([[0.0, 1.0]], (3, 1))

    actions = fit_candidate_actions(
        [first, second],
        n_directions=2,
        magnitudes=[0.1, 0.2],
        max_samples_per_corpus=3,
        random_state=4,
    )
    repeated = fit_candidate_actions(
        [first, second],
        n_directions=2,
        magnitudes=[0.1, 0.2],
        max_samples_per_corpus=3,
        random_state=4,
    )

    assert actions.shape == (5, 2)
    assert np.allclose(actions[0], 0.0)
    assert np.allclose(actions, repeated)
    assert np.allclose(np.linalg.norm(actions[1:3], axis=1), 0.1)
    assert np.allclose(np.linalg.norm(actions[3:], axis=1), 0.2)


def test_action_features_and_corrected_grid_use_query_major_order():
    queries = np.eye(2)
    actions = np.asarray([[0.0, 0.0], [0.0, 1.0]])
    components = np.eye(2)

    features = action_query_features(queries, actions, components)
    corrected = corrected_query_grid(queries, actions, components)

    assert features.shape == (4, 10)
    assert np.allclose(corrected[0], [1.0, 0.0])
    assert np.allclose(corrected[1], np.asarray([1.0, 1.0]) / np.sqrt(2.0))
    assert np.allclose(corrected[2], [0.0, 1.0])
    assert reshape_utilities(np.arange(4), 2, 2).tolist() == [[0, 1], [2, 3]]
