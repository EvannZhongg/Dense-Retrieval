import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_local_geometry_probe.py"
SPEC = importlib.util.spec_from_file_location("local_geometry_probe", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _row(lambda_, macro, ndcg, minimum):
    return {
        "lambda": lambda_,
        "macro_HitRate@10": macro,
        "macro_NDCG@10": ndcg,
        "min_HitRate@10_gain": minimum,
    }


def test_robust_selection_rejects_macro_gain_that_hurts_one_corpus():
    curve = [_row(0.0, 0.5, 0.4, 0.0), _row(0.2, 0.55, 0.45, -0.01)]

    assert MODULE._select_lambda(curve, "robust") == 0.0
    assert MODULE._select_lambda(curve, "macro") == 0.2


def test_robust_selection_accepts_strict_macro_gain_with_no_corpus_harm():
    curve = [
        _row(0.0, 0.5, 0.4, 0.0),
        _row(0.1, 0.52, 0.41, 0.0),
        _row(0.2, 0.52, 0.42, 0.0),
    ]

    assert MODULE._select_lambda(curve, "robust") == 0.2


def test_robust_selection_abstains_on_non_strict_macro_tie():
    curve = [_row(0.0, 0.5, 0.4, 0.0), _row(0.1, 0.5, 0.41, 0.0)]

    assert MODULE._select_lambda(curve, "robust") == 0.0


def test_macro_selection_preserves_smaller_lambda_tie_break():
    curve = [
        _row(0.0, 0.5, 0.4, 0.0),
        _row(0.1, 0.52, 0.41, 0.0),
        _row(0.2, 0.52, 0.45, 0.0),
    ]

    assert MODULE._select_lambda(curve, "macro") == 0.1
