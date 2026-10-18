import numpy as np
import pandas as pd

from compare_models import challenger_wins, evaluate

RULE = {"max_mae_increase": 0.01}


def metrics(mae):
    return {"mae": mae, "r2": 0.9, "mape": 0.11}


def test_better_challenger_wins():
    assert challenger_wins(metrics(60000), metrics(58000), RULE)


def test_tie_within_margin_wins():
    assert challenger_wins(metrics(60000), metrics(60500), RULE)


def test_worse_challenger_loses():
    assert not challenger_wins(metrics(60000), metrics(61000), RULE)


def test_evaluate_uses_the_same_rows_for_any_model():
    X = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
    y = pd.Series([100.0, 200.0, 300.0, 400.0])

    class Fixed:
        def __init__(self, factor):
            self.factor = factor

        def predict(self, X):
            return X["x"].to_numpy() * 100 * self.factor

    perfect, off = evaluate(Fixed(1.0), X, y), evaluate(Fixed(1.1), X, y)
    assert perfect["mae"] == 0 and perfect["r2"] == 1
    assert np.isclose(off["mape"], 0.1) and np.isclose(off["mae"], 25)
