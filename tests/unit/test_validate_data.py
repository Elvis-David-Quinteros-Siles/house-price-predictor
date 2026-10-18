import pandas as pd
import pytest

from validate_data import load_rules, validate

RULES = load_rules("configs/model_config.yaml")


def sales(months=("201501", "201502", "201503"), per_month=400, price=450_000):
    """Dataset válido: per_month ventas en cada mes, todas dentro del contrato."""
    base = dict(price=price, bedrooms=3, bathrooms=2, sqft_living=1800, sqft_lot=5000, yr_built=1975,
                condition=3, grade=7, view=0, waterfront=0, lat=47.66, long=-122.33)
    rows = [dict(base, id=f"{m}{i}", date=f"{m}15T000000") for m in months for i in range(per_month)]
    return pd.DataFrame(rows).astype(str)


def test_valid_dataset_has_no_problems():
    assert validate(sales(), RULES) == []


def test_missing_column_stops_the_checks():
    problems = validate(sales().drop(columns=["lat"]), RULES)
    assert problems == ["missing columns: lat"]


def test_empty_and_non_numeric_values():
    df = sales()
    df.loc[0, "price"] = None
    df.loc[1, "sqft_living"] = "mil"
    problems = " | ".join(validate(df, RULES))
    assert "price: 1 empty values" in problems and "sqft_living: 1 non-numeric" in problems


def test_values_outside_the_api_ranges():
    df = sales()
    df.loc[:2, "lat"] = "48.5"
    problems = validate(df, RULES)
    assert any(p.startswith("lat: 3 values outside") for p in problems)


def test_bad_date_format():
    df = sales()
    df.loc[0, "date"] = "2015-01-15"
    assert any("YYYYMMDDT000000" in p for p in validate(df, RULES))


def test_too_many_data_entry_errors():
    df = sales()
    df.loc[: len(df) // 10, "bedrooms"] = "0"
    assert any("cleaning would drop" in p for p in validate(df, RULES))


def test_month_published_twice():
    df = sales()
    doubled = pd.concat([df, df[df["date"].str.startswith("201503")]])
    assert any("repeated sales" in p for p in validate(doubled, RULES))


def test_gap_and_thin_months():
    problems = " | ".join(validate(sales(months=("201501", "201503")), RULES))
    assert "months without sales: 2015-02" in problems
    thin = sales()
    thin = thin[~thin["date"].str.startswith("201503") | (thin["id"].str[6:].astype(int) < 50)]
    assert "2015-03 (50)" in " | ".join(validate(thin, RULES))


@pytest.mark.parametrize("factor, flagged", [(1.1, False), (0.9, False), (2, True), (0.5, True)])
def test_latest_month_price_shift(factor, flagged):
    df = sales()
    df.loc[df["date"].str.startswith("201503"), "price"] = str(450_000 * factor)
    problems = validate(df, RULES)
    assert any("median price of 2015-03" in p for p in problems) == flagged
