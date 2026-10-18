"""
Pruebas de la API en marcha (contenedor construido con el modelo real).
API_URL apunta al servicio; por defecto http://localhost:8000.
"""
import os

import pytest
import requests

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")

# Casa promedio en Wallingford, Seattle.
HOUSE = {
    "sqft": 1900, "sqft_lot": 5000, "bedrooms": 3, "bathrooms": 2.0, "year_built": 1975,
    "condition": "Average", "grade": 7, "view": 0, "waterfront": False,
    "lat": 47.6615, "long": -122.3341,
}


def predict(**changes):
    return requests.post(f"{API_URL}/predict", json={**HOUSE, **changes}, timeout=10)


def price(**changes):
    response = predict(**changes)
    assert response.status_code == 200, response.text
    return response.json()["predicted_price"]


def test_health():
    response = requests.get(f"{API_URL}/health", timeout=5)
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy" and body["model_loaded"] is True
    assert body["model_version"]


def test_model_info():
    info = requests.get(f"{API_URL}/model-info", timeout=5).json()
    assert set(info["metrics"]) == {"mae", "r2", "mape"}
    assert info["model_name"] == "house_price_model" and info["model_version"]
    assert info["categories"]["condition"] == ["Average", "Fair", "Good", "Poor", "Very Good"]
    assert 0 < info["interval"]["low_ratio"] < 1 < info["interval"]["high_ratio"]
    assert info["map_area"]["max_distance_km"] > 0


def test_predict_response():
    body = predict().json()
    low, high = body["confidence_interval"]
    assert low < body["predicted_price"] < high
    assert body["latency_ms"] >= 0
    assert "location" in body["features_importance"]
    assert sum(body["features_importance"].values()) == pytest.approx(1, abs=0.01)


@pytest.mark.parametrize("cheaper, pricier", [
    (dict(sqft=1200), dict(sqft=3500)),
    (dict(grade=6), dict(grade=10)),
    # Auburn frente a Medina, de las zonas más baratas y más caras del condado.
    (dict(lat=47.3073, long=-122.2285), dict(lat=47.6205, long=-122.2340)),
])
def test_price_moves_in_expected_direction(cheaper, pricier):
    assert price(**cheaper) < price(**pricier)


def test_rejects_location_in_lake():
    response = predict(lat=47.6450, long=-122.2737)  # Lago Washington, junto al puente SR 520
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "location_too_far"


def test_rejects_unknown_condition():
    response = predict(condition="Excellent")
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "unknown_condition"


@pytest.mark.parametrize("field, value", [("lat", 48.5), ("bedrooms", 0), ("grade", 14), ("year_built", 2030)])
def test_rejects_out_of_range_values(field, value):
    assert predict(**{field: value}).status_code == 422


def test_batch_matches_single_predictions():
    houses = [HOUSE, {**HOUSE, "grade": 9}]
    batch = requests.post(f"{API_URL}/batch-predict", json=houses, timeout=10).json()
    assert batch == [price(), price(grade=9)]


def test_metrics_expose_drift():
    price()
    body = requests.get(f"{API_URL}/metrics", timeout=5).text
    samples = [line for line in body.splitlines() if line.startswith("house_price_drift_window_samples ")]
    assert samples and float(samples[0].split()[1]) >= 1
    assert 'house_price_feature_drift_psi{feature="sqft"}' in body
    assert 'house_price_feature_drift_psi{feature="predicted_price"}' in body
