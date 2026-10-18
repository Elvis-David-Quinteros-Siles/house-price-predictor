"""
Drift de datos: perfil de referencia del entrenamiento y PSI de producción.

train_model.py guarda con cada versión del modelo el perfil de sus datos de
entrenamiento (reference_profile.json, junto a model_metadata.json) y la API
compara con él las últimas casas que tasó. La API importa este archivo como
módulo `drift`.

PSI (population stability index) por variable, sobre deciles del entrenamiento:
  < 0.1 estable · 0.1-0.25 cambio moderado · > 0.25 cambio importante.
Con pocas muestras el PSI no baja a 0 aunque no haya drift: su valor esperado
es de unos (bins - 1) / n, 0.09 con 100 casas y 10 bins.
"""
import numpy as np

BINS = 10
# waterfront (0/1), view y grade son discretas: los deciles repetidos se funden
# y quedan menos bins, uno por valor frecuente.
NUMERIC_FEATURES = ['sqft', 'sqft_lot', 'bedrooms', 'bathrooms', 'year_built', 'grade', 'view',
                    'waterfront', 'lat', 'long']
CATEGORICAL_FEATURES = ['condition']
# Drift de la salida: precio predicho frente a las predicciones fuera de muestra de la CV.
PREDICTION = 'predicted_price'


def _numeric_bins(edges, values):
    # side="left": cada bin es (e[i-1], e[i]], así 0 y 1 caen en bins distintos
    # cuando el único borde es 0 (waterfront).
    return np.searchsorted(edges, np.asarray(values, dtype=float), side="left")


def numeric_profile(values) -> dict:
    values = np.asarray(values, dtype=float)
    edges = np.unique(np.quantile(values, np.linspace(0, 1, BINS + 1)[1:-1]))
    counts = np.bincount(_numeric_bins(edges, values), minlength=len(edges) + 1)
    return {"type": "numeric", "edges": edges.tolist(), "proportions": (counts / counts.sum()).tolist()}


def categorical_profile(values) -> dict:
    categories, counts = np.unique(np.asarray(values).astype(str), return_counts=True)
    return {"type": "categorical", "categories": categories.tolist(),
            "proportions": (counts / counts.sum()).tolist()}


def build_profile(X, predictions) -> dict:
    """Perfil de las entradas de entrenamiento (X) y de las predicciones de referencia."""
    features = {c: numeric_profile(X[c]) for c in NUMERIC_FEATURES}
    features.update({c: categorical_profile(X[c]) for c in CATEGORICAL_FEATURES})
    features[PREDICTION] = numeric_profile(predictions)
    return {"bins": BINS, "rows": int(len(X)), "features": features}


def bin_counts(feature: dict, values) -> np.ndarray:
    if feature["type"] == "numeric":
        edges = np.asarray(feature["edges"])
        return np.bincount(_numeric_bins(edges, values), minlength=len(edges) + 1)
    # La API rechaza las categorías que el modelo no conoce: aquí no llegan.
    index = {c: i for i, c in enumerate(feature["categories"])}
    known = [index[str(v)] for v in values if str(v) in index]
    return np.bincount(np.asarray(known, dtype=int), minlength=len(index))


def psi(feature: dict, values) -> float:
    """PSI de `values` frente a la distribución de referencia de `feature`."""
    counts = bin_counts(feature, values)
    # Suavizado de Laplace (+0.5 por bin): con pocas casas hay bins vacíos y
    # ln(0) haría el PSI infinito.
    actual = (counts + 0.5) / (counts.sum() + 0.5 * len(counts))
    expected = np.maximum(np.asarray(feature["proportions"]), 1e-4)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def drift_scores(profile: dict, rows: list[dict]) -> dict:
    """PSI de cada variable del perfil sobre `rows` (entradas + predicted_price)."""
    return {name: round(psi(feature, [row[name] for row in rows]), 4)
            for name, feature in profile["features"].items()}
