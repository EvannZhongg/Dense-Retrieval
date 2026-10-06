import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_coefficient_predictability.py"
SPEC = importlib.util.spec_from_file_location("coefficient_predictability", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_linear_model_recovers_predictable_coefficients():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(80, 4))
    true_w = rng.normal(size=(4, 3))
    true_b = rng.normal(size=3)
    y = x @ true_w + true_b
    ids = [f"q{i}" for i in range(len(x))]
    model, validation = MODULE.select_and_fit_linear_model(
        x, y, ids, alphas=[1e-8, 1.0]
    )

    assert validation.loc[validation.validation_mse.idxmin(), "alpha"] == 1e-8
    assert np.allclose(model.predict(x), y, atol=1e-6)


def test_predicted_delta_reconstruction():
    class Model:
        def predict(self, x):
            return np.tile(np.array([[2.0, -1.0]]), (len(x), 1))

    mean = np.array([1.0, 1.0, 1.0])
    components = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    result = MODULE.predicted_delta(np.zeros((2, 3)), Model(), mean, components)
    assert np.allclose(result, [[3.0, 0.0, 1.0], [3.0, 0.0, 1.0]])


def test_regression_split_is_disjoint_and_stable():
    ids = [f"q{i}" for i in range(20)]
    fit_a, validation_a = MODULE.stable_regression_split(ids)
    fit_b, validation_b = MODULE.stable_regression_split(ids)
    assert not np.any(fit_a & validation_a)
    assert np.array_equal(fit_a, fit_b)
    assert np.array_equal(validation_a, validation_b)
