import json
import time
import joblib
import pandas as pd
from datetime import datetime
from schemas import HousePredictionRequest, PredictionResponse

# Load model, preprocessor and training metadata
MODEL_PATH = "models/trained/house_price_model.pkl"
PREPROCESSOR_PATH = "models/trained/preprocessor.pkl"
METADATA_PATH = "models/trained/model_metadata.json"

try:
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    with open(METADATA_PATH) as f:
        metadata = json.load(f)
except Exception as e:
    raise RuntimeError(f"Error loading model, preprocessor or metadata: {str(e)}")

# Categorías vistas en entrenamiento (zipcodes de King County y estados).
_encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
CATEGORIES = {
    column: sorted(str(c) for c in values)
    for column, values in zip(preprocessor.named_transformers_["cat"].feature_names_in_, _encoder.categories_)
}

def _compute_features_importance() -> dict:
    """
    Global feature importance of the model, grouped by preprocessor input column
    (the one-hot columns of location/condition are summed back together).
    """
    # Con log_target el modelo es un TransformedTargetRegressor.
    importances = getattr(getattr(model, "regressor_", model), "feature_importances_", None)
    if importances is None:
        return {}

    grouped = {}
    for name, value in zip(preprocessor.get_feature_names_out(), importances):
        transformer, column = name.split("__", 1)
        if transformer == "cat":
            column = next(c for c in preprocessor.named_transformers_["cat"].feature_names_in_
                          if column.startswith(f"{c}_"))
        grouped[column] = grouped.get(column, 0.0) + float(value)

    total = sum(grouped.values()) or 1.0
    return {k: round(v / total, 4) for k, v in sorted(grouped.items(), key=lambda kv: kv[1], reverse=True)}

FEATURES_IMPORTANCE = _compute_features_importance()

def model_info() -> dict:
    """Metadata of the served model plus the values the inputs accept."""
    return {**metadata, "features_importance": FEATURES_IMPORTANCE, "categories": CATEGORIES}

def _prepare(requests: list[HousePredictionRequest]) -> pd.DataFrame:
    input_data = pd.DataFrame([req.dict() for req in requests])
    for column, allowed in CATEGORIES.items():
        unknown = set(input_data[column]) - set(allowed)
        if unknown:
            raise ValueError(f"Unknown {column}: {', '.join(sorted(unknown))}. See /model-info for valid values.")
    input_data['bed_bath_ratio'] = input_data['bedrooms'] / input_data['bathrooms']
    return preprocessor.transform(input_data)

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
