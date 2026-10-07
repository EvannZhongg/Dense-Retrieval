import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_local_geometry_gate_loco.py"
SPEC = importlib.util.spec_from_file_location("local_geometry_gate_loco", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_gated_queries_select_rows_without_changing_others():
    original = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    corrected = np.asarray([[0.8, 0.2], [0.2, 0.8]], dtype=np.float32)

    result = MODULE._gated_queries(original, corrected, [True, False])

    assert np.array_equal(result[0], corrected[0])
    assert np.array_equal(result[1], original[1])


def test_threshold_candidates_include_apply_all_and_abstain_all():
    values = MODULE._thresholds(np.asarray([-0.2, 0.0, 0.4]))

    assert values[0] == -np.inf
    assert values[-1] == np.inf
    assert 0.0 in values
