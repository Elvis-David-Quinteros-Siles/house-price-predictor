from engineer import create_features, create_preprocessor
from run_processing import clean_data


def test_bed_bath_ratio(raw_sales):
    featured = create_features(clean_data(raw_sales))
    assert (featured["bed_bath_ratio"] == featured["bedrooms"] / featured["bathrooms"]).all()


def test_no_target_leakage(raw_sales):
    featured = create_features(clean_data(raw_sales))
    # price_per_sqft se deriva del precio: no debe existir como feature.
    assert "price_per_sqft" not in featured.columns
    preprocessor = create_preprocessor().fit(featured.drop(columns=["price"]))
    assert not any("price" in name for name in preprocessor.get_feature_names_out())


def test_preprocessor_output(raw_sales):
    featured = create_features(clean_data(raw_sales))
    preprocessor = create_preprocessor()
    transformed = preprocessor.fit_transform(featured.drop(columns=["price"]))
    names = list(preprocessor.get_feature_names_out())
    assert transformed.shape == (len(featured), len(names))
    assert {"num__lat", "num__long", "num__grade"} <= set(names)
    # La salida es densa para poder guardarse como CSV.
    assert hasattr(transformed, "shape") and not hasattr(transformed, "toarray")
