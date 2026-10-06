import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts" / "summarize_oracle_correction.py"
SPEC = importlib.util.spec_from_file_location("summarize_oracle_correction", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_apply_oracle_correction_moves_only_available_queries_and_normalizes():
    queries = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    positives = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    available = np.array([True, False])

    corrected = MODULE.apply_oracle_correction(
        queries, positives, available, lambda_=0.5
    )

    assert np.allclose(corrected[0], [2**-0.5, 2**-0.5])
    assert np.allclose(corrected[1], queries[1])
    assert np.allclose(np.linalg.norm(corrected, axis=1), 1.0)
