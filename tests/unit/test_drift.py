import numpy as np
import pandas as pd
import pytest

from drift import BINS, PREDICTION, build_profile, drift_scores, numeric_profile, psi
from prediction_log import DriftMonitor


@pytest.fixture
def training():
    rng = np.random.default_rng(0)
    n = 5000
    X = pd.DataFrame({
        "sqft": rng.lognormal(7.5, 0.4, n), "sqft_lot": rng.lognormal(8.9, 0.8, n),
        "bedrooms": rng.integers(1, 7, n), "bathrooms": rng.integers(1, 5, n) / 1.0,
        "year_built": rng.integers(1900, 2016, n), "grade": rng.integers(5, 11, n),
        "view": rng.choice([0, 0, 0, 0, 0, 0, 0, 0, 0, 2], n), "waterfront": (rng.random(n) < 0.01).astype(int),
        "lat": rng.uniform(47.2, 47.7, n), "long": rng.uniform(-122.4, -121.8, n),
        "condition": rng.choice(["Average", "Good", "Very Good"], n, p=[0.65, 0.26, 0.09]),
    })
    return X, rng.lognormal(13, 0.5, n)


def rows_from(X, prices):
    return [{**row, PREDICTION: p} for row, p in zip(X.to_dict("records"), prices)]


def test_same_distribution_is_stable(training):
    X, prices = training
    scores = drift_scores(build_profile(X.iloc[:4000], prices[:4000]),
                          rows_from(X.iloc[4000:], prices[4000:]))
    assert set(scores) == set(X.columns) | {PREDICTION}
    assert max(scores.values()) < 0.1


def test_shifted_feature_is_flagged(training):
    X, prices = training
    profile = build_profile(X, prices)
    bigger = X.assign(sqft=X["sqft"] * 1.6)
    scores = drift_scores(profile, rows_from(bigger.iloc[:500], prices[:500]))
    assert scores["sqft"] > 0.25
    assert scores["lat"] < 0.1


def test_binary_feature_keeps_both_values_apart():
    feature = numeric_profile([0] * 99 + [1])
    assert feature["edges"] == [0.0]
    assert psi(feature, [1] * 50) > 1


def test_categorical_shift_is_flagged(training):
    X, prices = training
    profile = build_profile(X, prices)
    assert psi(profile["features"]["condition"], ["Very Good"] * 200) > 0.25


def test_numeric_profile_uses_deciles():
    feature = numeric_profile(np.arange(1000))
    assert len(feature["proportions"]) == BINS
    assert np.allclose(feature["proportions"], 0.1, atol=0.002)


def test_monitor_survives_restart(training, tmp_path):
    X, prices = training
    profile = build_profile(X, prices)
    monitor = DriftMonitor(profile, "1", log_dir=tmp_path, window=3)
    houses = X.iloc[:5].to_dict("records")
    monitor.observe(houses[:2], list(prices[:2]))
    monitor.observe(houses[2:], list(prices[2:5]))

    restarted = DriftMonitor(profile, "2", log_dir=tmp_path, window=3)
    assert [r[PREDICTION] for r in restarted.recent] == pytest.approx(prices[2:5])
    assert restarted.recent[0]["model_version"] == "1"


def test_monitor_without_profile_only_logs(tmp_path):
    monitor = DriftMonitor(None, "1", log_dir=tmp_path, window=10)
    monitor.observe([{"sqft": 1000.0}], [300000.0])
    assert len(list(tmp_path.glob("predictions-*.jsonl"))) == 1
