"""
Métricas propias del modelo, junto a las HTTP de prometheus-fastapi-instrumentator.
Se exponen en el mismo /metrics.
"""
from prometheus_client import Counter, Gauge, Histogram

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
