import json
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree
from datetime import datetime
from engineer import EARTH_RADIUS_KM, INPUT_FEATURES
from schemas import HousePredictionRequest, PredictionResponse

# Load model (full pipeline: features + preprocessor + regressor) and training metadata
MODEL_PATH = "models/trained/house_price_model.pkl"
METADATA_PATH = "models/trained/model_metadata.json"
LOCATIONS_PATH = "models/trained/sold_locations.npy"

# Distancia máxima a la venta más cercana para aceptar una ubicación.
# El 99 % de las ventas tiene otra a menos de 0,55 km; con 1 km se aceptaban
# puntos dentro del lago Washington cerca de la orilla.
MAX_DISTANCE_KM = 0.5

try:
    model = joblib.load(MODEL_PATH)
    with open(METADATA_PATH) as f:
        metadata = json.load(f)
    sold_locations = np.load(LOCATIONS_PATH)
except Exception as e:
    raise RuntimeError(f"Error loading model or metadata: {str(e)}")

# El modelo es un TransformedTargetRegressor (log del precio) sobre el pipeline.
_pipeline = model.regressor_
preprocessor = _pipeline.named_steps["preprocessor"]

# Índice de las casas vendidas para validar la ubicación pedida.
_locations_tree = BallTree(np.radians(sold_locations), metric="haversine")
MAP_AREA = {
    "center": [float(v) for v in sold_locations.mean(axis=0)],
    "bounds": [[float(v) for v in sold_locations.min(axis=0)], [float(v) for v in sold_locations.max(axis=0)]],
    "max_distance_km": MAX_DISTANCE_KM,
}

class InvalidInputError(ValueError):
    """Input the model cannot price; `code` lets clients show their own message."""
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code

# Categorías vistas en entrenamiento (estados de la casa).
_encoder = preprocessor.named_transformers_["cat"]
CATEGORIES = {
    column: sorted(str(c) for c in values)
    for column, values in zip(preprocessor.named_transformers_["cat"].feature_names_in_, _encoder.categories_)
}

def _compute_features_importance() -> dict:
    """
    Global feature importance of the model, grouped by preprocessor input column
    (the one-hot columns of condition are summed back together, and lat/long and
    the neighbour prices are shown as "location").
    """
    importances = getattr(_pipeline.named_steps["model"], "feature_importances_", None)
    if importances is None:
        return {}

    grouped = {}
    for name, value in zip(preprocessor.get_feature_names_out(), importances):
        transformer, column = name.split("__", 1)
        if column == "sale_month":
            # Al predecir siempre vale el último mes: no explica una predicción.
            continue
        if column in ("lat", "long") or column.startswith("nb_"):
            column = "location"
        elif transformer == "cat":
            column = next(c for c in preprocessor.named_transformers_["cat"].feature_names_in_
                          if column.startswith(f"{c}_"))
        grouped[column] = grouped.get(column, 0.0) + float(value)

    total = sum(grouped.values()) or 1.0
    return {k: round(v / total, 4) for k, v in sorted(grouped.items(), key=lambda kv: kv[1], reverse=True)}

FEATURES_IMPORTANCE = _compute_features_importance()

def model_info() -> dict:
    """Metadata of the served model plus the values the inputs accept."""
    return {**metadata, "features_importance": FEATURES_IMPORTANCE, "categories": CATEGORIES, "map_area": MAP_AREA}

def _prepare(requests: list[HousePredictionRequest]) -> pd.DataFrame:
    input_data = pd.DataFrame([req.dict() for req in requests])
    input_data['waterfront'] = input_data['waterfront'].astype(int)
    for column, allowed in CATEGORIES.items():
        unknown = set(input_data[column]) - set(allowed)
        if unknown:
            raise InvalidInputError(
                f"unknown_{column}",
                f"Unknown {column}: {', '.join(sorted(unknown))}. See /model-info for valid values.",
            )

    distances, _ = _locations_tree.query(np.radians(input_data[["lat", "long"]].to_numpy()), k=1)
    too_far = distances[:, 0] * EARTH_RADIUS_KM > MAX_DISTANCE_KM
    if too_far.any():
        raise InvalidInputError(
            "location_too_far",
            f"Location too far from any house sold in King County (more than {MAX_DISTANCE_KM:g} km): "
            "pick a point on land inside the county."
        )
    return input_data[INPUT_FEATURES]

def predict_price(request: HousePredictionRequest) -> PredictionResponse:
    """
    Predict house price based on input features.
    """
    start = time.perf_counter()
    predicted_price = round(float(model.predict(_prepare([request]))[0]), 2)

    # Intervalo empírico: el 90% de las casas de test cayó en este rango relativo.
    interval = metadata["interval"]
    confidence_interval = [
        round(predicted_price * interval["low_ratio"], 2),
        round(predicted_price * interval["high_ratio"], 2),
    ]

    return PredictionResponse(
        predicted_price=predicted_price,
        confidence_interval=confidence_interval,
        features_importance=FEATURES_IMPORTANCE,
        prediction_time=datetime.now().isoformat(),
        latency_ms=round((time.perf_counter() - start) * 1000, 2),
    )

def batch_predict(requests: list[HousePredictionRequest]) -> list[float]:
    """
    Perform batch predictions.
    """
    predictions = model.predict(_prepare(requests))
    return [round(float(p), 2) for p in predictions]
