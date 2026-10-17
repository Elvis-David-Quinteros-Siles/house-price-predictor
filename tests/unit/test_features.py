import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor

from engineer import INPUT_FEATURES, HouseFeatures, build_pipeline, create_features
from run_processing import clean_data


def houses(n=30, seed=0):
    """Synthetic sales spread over Seattle, in the cleaned schema."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "sqft": rng.integers(800, 4000, n), "sqft_lot": rng.integers(2000, 20000, n),
        "bedrooms": rng.integers(1, 6, n), "bathrooms": rng.choice([1.0, 1.5, 2.0, 3.0], n),
        "year_built": rng.integers(1900, 2015, n),
        "condition": rng.choice(["Average", "Good", "Very Good"], n),
        "grade": rng.integers(5, 12, n), "view": rng.integers(0, 5, n), "waterfront": rng.integers(0, 2, n),
        "lat": rng.uniform(47.5, 47.7, n), "long": rng.uniform(-122.4, -122.2, n),
        "sale_month": rng.integers(5, 18, n),
    })


def test_bed_bath_ratio(raw_sales):
    featured = create_features(clean_data(raw_sales))
    assert (featured["bed_bath_ratio"] == featured["bedrooms"] / featured["bathrooms"]).all()


def test_sale_month(raw_sales):
    cleaned = clean_data(raw_sales)
    # Junio de 2014 -> 6; marzo de 2015 -> 15.
    assert set(cleaned["sale_month"]) <= set(range(1, 25))
    assert 15 in set(cleaned["sale_month"])


def test_neighbours_exclude_the_house_itself():
    X = houses()
    log_price = np.arange(len(X), dtype=float)
    featured = HouseFeatures(k=1).fit_transform(X, log_price)
    # Con k=1 el único vecino es otra casa: nunca su propio precio.
    assert (featured["nb_log_price"].to_numpy() != log_price).all()


def test_prediction_uses_latest_sale_month():
    X = houses()
    features = HouseFeatures(k=3).fit(X, np.ones(len(X)))
    new_house = X[INPUT_FEATURES].head(1)
    assert features.transform(new_house)["sale_month"].iloc[0] == X["sale_month"].max()


def test_no_target_leakage():
    X = houses()
    pipeline = build_pipeline(DummyRegressor(), neighbours=3).fit(X, np.ones(len(X)))
    names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    # price_per_sqft se deriva del precio: no debe existir como feature.
    assert not any("price_per_sqft" in name for name in names)


def test_pipeline_predicts_from_api_inputs():
    X = houses()
    pipeline = build_pipeline(DummyRegressor(), neighbours=3).fit(X, np.log(X["sqft"] * 300.0))
    names = list(pipeline.named_steps["preprocessor"].get_feature_names_out())
    assert {"num__lat", "num__long", "num__grade", "num__nb_log_price"} <= set(names)
    # La API envía solo INPUT_FEATURES, sin fecha de venta.
    assert pipeline.predict(X[INPUT_FEATURES].head(5)).shape == (5,)
