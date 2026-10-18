"""
Métricas propias del modelo, junto a las HTTP de prometheus-fastapi-instrumentator.
Se exponen en el mismo /metrics.
"""
import logging
import os
from pathlib import Path

from performance import REAL_ERROR_FILE, read_real_error
from prometheus_client import REGISTRY, Counter, Gauge, Histogram
from prometheus_client.core import GaugeMetricFamily

logger = logging.getLogger(__name__)

PREDICTIONS = Counter(
    "house_price_predictions_total",
    "Houses priced, by endpoint and outcome (ok or the rejection code).",
    ["endpoint", "outcome"],
)

PREDICTED_PRICE = Histogram(
    "house_price_predicted_price_dollars",
    "Distribution of predicted prices; a shift here is the first sign of drift.",
    buckets=(150_000, 250_000, 350_000, 450_000, 550_000, 650_000, 800_000,
             1_000_000, 1_500_000, 2_000_000, 3_000_000, 5_000_000),
)

MODEL_TEST_METRIC = Gauge(
    "house_price_model_test_metric",
    "Hold-out test metrics of the model being served.",
    ["metric"],
)

MODEL_INFO = Gauge(
    "house_price_model_info",
    "Model being served (value is always 1; details in the labels).",
    ["version", "algorithm", "train_rows"],
)


# Resultados posibles: se crean a 0 al arrancar porque Prometheus no cuenta como
# incremento la primera muestra de una serie (se perderían las primeras peticiones).
OUTCOMES = ("ok", "location_too_far", "unknown_condition")
for _endpoint in ("predict", "batch"):
    for _outcome in OUTCOMES:
        PREDICTIONS.labels(endpoint=_endpoint, outcome=_outcome)


FEATURE_DRIFT = Gauge(
    "house_price_feature_drift_psi",
    "PSI of each input (and of the predicted price) over the last predictions, "
    "against the training data of the served model.",
    ["feature"],
)

DRIFT_WINDOW_SAMPLES = Gauge(
    "house_price_drift_window_samples",
    "Predictions the drift PSI is computed over (the last DRIFT_WINDOW).",
)


def publish_model_metadata(metadata: dict) -> None:
    for name, value in metadata["metrics"].items():
        MODEL_TEST_METRIC.labels(metric=name).set(value)
    MODEL_INFO.labels(version=str(metadata.get("model_version", "unknown")), algorithm=metadata["algorithm"],
                      train_rows=str(metadata["train_rows"])).set(1)


def record(endpoint: str, prices: list[float]) -> None:
    PREDICTIONS.labels(endpoint=endpoint, outcome="ok").inc(len(prices))
    for price in prices:
        PREDICTED_PRICE.observe(price)


def record_rejection(endpoint: str, code: str) -> None:
    PREDICTIONS.labels(endpoint=endpoint, outcome=code).inc()


def publish_drift(samples: int, scores: dict) -> None:
    DRIFT_WINDOW_SAMPLES.set(samples)
    for feature, value in scores.items():
        FEATURE_DRIFT.labels(feature=feature).set(value)


# Error real del @champion con las últimas ventas publicadas: lo calcula
# scripts/production_performance.py en el job de entrenamiento continuo y lo
# deja en un volumen que la API monta. Se relee en cada scrape porque el job
# lo escribe sin reiniciar la API (solo la reinicia si promueve otra versión).
REAL_ERROR_PATH = Path(os.getenv("REAL_ERROR_DIR", "data/performance")) / REAL_ERROR_FILE

REAL_ERROR_METRICS = (
    # (nombre, clave en el JSON, descripción)
    ("house_price_real_mae_dollars", "mae", "MAE against the sale prices of the last published month."),
    ("house_price_real_mape", "mape", "MAPE against the sale prices of the last published month."),
    ("house_price_real_bias", "bias",
     "Median of (predicted - sold) / sold: negative when the model prices below the market."),
    ("house_price_real_interval_coverage", "interval_coverage",
     "Share of those sales inside the 90% interval the API returns."),
    ("house_price_real_sales", "sales", "Sales the real error is computed over."),
    ("house_price_real_mape_limit", "mape_limit", "Real MAPE above which the model counts as degraded."),
    ("house_price_real_max_abs_bias", "max_abs_bias", "Absolute bias above which the model counts as degraded."),
    ("house_price_real_degraded", "degraded", "1 if the evaluated model is degraded on those sales."),
)


class RealErrorCollector:
    """Última evaluación con ventas reales, con la versión evaluada y el mes en las etiquetas."""

    def __init__(self, path: Path):
        self.path = path

    def collect(self):
        try:
            evaluation = read_real_error(self.path)
        except (OSError, ValueError) as e:
            logger.warning(f"Could not read the real error ({self.path}): {e}")
            return
        if evaluation is None:
            return
        labels = ["model_version", "month"]
        values = [str(evaluation["model_version"]), evaluation["month"]]
        for name, key, documentation in REAL_ERROR_METRICS:
            family = GaugeMetricFamily(name, documentation, labels=labels)
            family.add_metric(values, float(evaluation[key]))
            yield family

    def describe(self):
        # Sin describe() el registro llamaría a collect() al registrarlo.
        return []


REGISTRY.register(RealErrorCollector(REAL_ERROR_PATH))
