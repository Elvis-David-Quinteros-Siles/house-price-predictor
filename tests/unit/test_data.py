from run_processing import CONDITION_LABELS, clean_data


def test_keeps_only_latest_sale_of_each_house(raw_sales):
    cleaned = clean_data(raw_sales)
    # La casa 1 aparece una vez, con el precio de la venta de 2015.
    assert sorted(cleaned["price"]) == [450000, 500000]


def test_drops_impossible_bedrooms_and_bathrooms(raw_sales):
    cleaned = clean_data(raw_sales)
    assert cleaned["bedrooms"].between(1, 10).all()
    assert (cleaned["bathrooms"] > 0).all()
    assert len(cleaned) == 2


def test_maps_to_project_schema(raw_sales):
    cleaned = clean_data(raw_sales)
    assert list(cleaned.columns) == [
        "price", "sqft", "sqft_lot", "bedrooms", "bathrooms", "year_built",
        "condition", "grade", "view", "waterfront", "lat", "long", "sale_month",
    ]
    assert set(cleaned["condition"]) <= set(CONDITION_LABELS.values())
    assert cleaned.notnull().all().all()
