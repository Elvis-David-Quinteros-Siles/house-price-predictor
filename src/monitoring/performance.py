"""
Error real del modelo: lo que tasó frente a lo que se vendió.

El drift (drift.py) solo dice que las entradas cambiaron. Para saber si el
modelo acierta hace falta el precio real, que llega después: cuando se publica
un mes de ventas, el @champion tasa esas casas (ninguna estaba en su
entrenamiento) y se compara con el precio de venta.
scripts/production_performance.py lo hace antes de reentrenar y la API publica
el resultado en /metrics (metrics.py). La API importa este archivo como módulo
`performance`.
"""
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score

# Lo que deja scripts/production_performance.py y lee la API (un volumen compartido).
REAL_ERROR_FILE = "real-error.json"


def month_label(sale_month: int) -> str:
    """sale_month (1 = enero 2014) como AAAA-MM, el formato del tag data_until."""
    year, month = divmod(int(sale_month) - 1, 12)
    return f"{2014 + year}-{month + 1:02d}"


def month_index(label: str) -> int:
    """AAAA-MM como sale_month (1 = enero 2014)."""
    year, month = (int(part) for part in label.split("-"))
    return (year - 2014) * 12 + month


def unseen_months(sale_months, data_until: str) -> list[int]:
    """Meses de venta posteriores al último con el que se entrenó el modelo."""
    last = month_index(data_until)
    return sorted(int(m) for m in set(sale_months) if m > last)


def evaluate_sales(y_true, y_pred, interval: dict) -> dict:
    """Error de las tasaciones `y_pred` frente a los precios de venta `y_true`.

    bias es la mediana de (predicho - real) / real: negativa si el modelo tasa
    por debajo del mercado, lo que pasa cuando los precios suben después de
    entrenar (el MAPE solo dice cuánto se equivoca, no hacia dónde).
    interval_coverage: ventas dentro del intervalo del 90 % que da la API.
    """
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    ratio = y_true / y_pred
    return {
        "sales": int(len(y_true)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mape": float(mean_absolute_percentage_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "bias": float(np.median(y_pred / y_true - 1)),
        "interval_coverage": float(np.mean((ratio >= interval["low_ratio"]) & (ratio <= interval["high_ratio"]))),
    }


def mape_limit(test_mape: float, rule: dict) -> float:
    """MAPE real a partir del cual el modelo se considera degradado."""
    return test_mape * (1 + rule["max_mape_increase"])


def degradation(result: dict, limit: float, rule: dict) -> list[str]:
    """Motivos por los que el modelo está degradado (vacío si no lo está).

    Con pocas ventas el MAPE y el sesgo son ruido: no se declara degradado.
    """
    if result["sales"] < rule["min_sales"]:
        return []
    reasons = []
    if result["mape"] > limit:
        reasons.append(f"MAPE {result['mape']:.2%} > {limit:.2%}")
    if abs(result["bias"]) > rule["max_abs_bias"]:
        reasons.append(f"sesgo {result['bias']:+.1%} (máx. ±{rule['max_abs_bias']:.0%})")
    return reasons


def read_real_error(path: Path) -> dict | None:
    """Última evaluación con ventas reales, o None si todavía no hay ninguna."""
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return None
