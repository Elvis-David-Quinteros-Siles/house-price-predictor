import json

import numpy as np
import pytest

from performance import degradation, evaluate_sales, mape_limit, month_index, month_label, unseen_months

INTERVAL = {"low_ratio": 0.8, "high_ratio": 1.2}
RULE = {"max_mape_increase": 0.15, "max_abs_bias": 0.05, "min_sales": 200}


def test_month_label_and_index_round_trip():
    assert month_label(1) == "2014-01" and month_label(15) == "2015-03"
    assert all(month_index(month_label(m)) == m for m in range(1, 30))


def test_unseen_months_are_the_ones_after_training():
    assert unseen_months([13, 14, 14, 15, 16], "2015-02") == [15, 16]
    assert unseen_months([13, 14], "2015-02") == []


def test_exact_predictions_have_no_error():
    prices = [300_000, 500_000, 900_000]
    result = evaluate_sales(prices, prices, INTERVAL)
    assert result["sales"] == 3
    assert result["mae"] == 0 and result["mape"] == 0 and result["bias"] == 0
    assert result["interval_coverage"] == 1


def test_bias_is_negative_when_the_market_went_up():
    sold = np.array([300_000, 500_000, 900_000, 400_000])
    result = evaluate_sales(sold * 1.1, sold, INTERVAL)
    assert result["bias"] == pytest.approx(1 / 1.1 - 1)
    assert result["mape"] == pytest.approx(1 - 1 / 1.1)


def test_coverage_counts_sales_inside_the_interval():
    # Real/predicho: 1.0 y 1.1 dentro de [0.8, 1.2]; 1.5 y 0.5 fuera.
    result = evaluate_sales([100, 110, 150, 50], [100, 100, 100, 100], INTERVAL)
    assert result["interval_coverage"] == 0.5


def evaluation(**overrides):
    return {"sales": 1000, "mape": 0.10, "bias": -0.01, **overrides}


def test_degradation_reasons():
    limit = mape_limit(0.10, RULE)
    assert limit == pytest.approx(0.115)
    assert degradation(evaluation(), limit, RULE) == []
    assert len(degradation(evaluation(mape=0.12), limit, RULE)) == 1
    assert len(degradation(evaluation(bias=-0.06), limit, RULE)) == 1
    assert len(degradation(evaluation(mape=0.12, bias=0.06), limit, RULE)) == 2


def test_few_sales_are_not_degradation():
    assert degradation(evaluation(sales=50, mape=0.5, bias=-0.3), 0.115, RULE) == []


def test_collector_publishes_the_last_evaluation(tmp_path):
    from metrics import RealErrorCollector

    path = tmp_path / "real-error.json"
    collector = RealErrorCollector(path)
    assert list(collector.collect()) == []

    path.write_text(json.dumps({"model_version": "4", "month": "2015-03", "sales": 1875, "mae": 63316.0,
                                "mape": 0.1194, "r2": 0.88, "bias": -0.035, "interval_coverage": 0.884,
                                "test_mape": 0.1083, "mape_limit": 0.1245, "max_abs_bias": 0.05,
                                "degraded": False, "reasons": []}))
    families = {f.name: f for f in collector.collect()}
    assert families["house_price_real_mape"].samples[0].value == 0.1194
    assert families["house_price_real_mape"].samples[0].labels == {"model_version": "4", "month": "2015-03"}
    assert families["house_price_real_degraded"].samples[0].value == 0


def test_collector_survives_a_broken_file(tmp_path):
    from metrics import RealErrorCollector

    path = tmp_path / "real-error.json"
    path.write_text("{")
    assert list(RealErrorCollector(path).collect()) == []
