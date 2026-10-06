import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts" / "pca_delta_variance.py"
SPEC = importlib.util.spec_from_file_location("pca_delta_variance", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_pca_variance_spectrum_is_cumulative_and_complete():
    samples = np.array(
        [[-2.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
        dtype=np.float64,
    )
    _, ratio, cumulative = MODULE.pca_variance_spectrum(samples)

    assert np.allclose(ratio, [1.0, 0.0])
    assert np.all(np.diff(cumulative) >= 0)
    assert np.isclose(cumulative[-1], 1.0)


def test_query_level_delta_uses_closest_positive_and_unit_tangent_direction():
    class Sample:
        def __init__(self, query_id):
            self.query_id = query_id

    class Dataset:
        queries = [Sample("q1"), Sample("q2")]
        corpus = {"d1": object(), "d2": object()}
        qrels = {"q1": {"d1": 1, "d2": 2}, "q2": {"missing": 1}}

    queries = np.array([[1.0, 0.0], [0.0, 1.0]])
    documents = np.array([[0.5, 0.0], [1.0, 3.0]])
    variants, missing = MODULE.build_query_level_deltas(
        Dataset(), queries, documents
    )

    assert np.allclose(variants["delta_star"], [[0.0, 3.0]])
    assert np.allclose(variants["delta_star_q_orthogonal_unit"], [[0.0, 1.0]])
    assert missing == [("q2", "missing")]
