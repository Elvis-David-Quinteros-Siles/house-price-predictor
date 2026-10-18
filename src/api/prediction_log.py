"""
Registro de predicciones y drift de las entradas.

Cada casa tasada se añade a un JSONL mensual en PREDICTION_LOG_DIR (un volumen
en el despliegue) y el PSI de cada variable se recalcula sobre las últimas
DRIFT_WINDOW frente al perfil de entrenamiento del modelo servido. Al arrancar
se releen las últimas del registro: el drift no vuelve a cero con cada
reinicio, y el registro queda como datos de producción para reentrenar.
"""
import json
import logging
import os
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import metrics
from drift import PREDICTION, drift_scores

logger = logging.getLogger(__name__)

PREDICTION_LOG_DIR = Path(os.getenv("PREDICTION_LOG_DIR", "data/predictions"))
DRIFT_WINDOW = int(os.getenv("DRIFT_WINDOW", "500"))


class DriftMonitor:
    def __init__(self, profile: dict | None, model_version: str | None,
                 log_dir: Path = PREDICTION_LOG_DIR, window: int = DRIFT_WINDOW):
        # Sin perfil (modelo registrado antes del monitoreo) solo se registran las predicciones.
        self.profile = profile
        self.model_version = model_version
        self.log_dir = log_dir
        self.recent = deque(self._read_recent(window), maxlen=window)
        self._publish()

    def _read_recent(self, window: int) -> list[dict]:
        rows = []
        for path in sorted(self.log_dir.glob("predictions-*.jsonl"), reverse=True):
            rows = [json.loads(line) for line in path.read_text().splitlines() if line] + rows
            if len(rows) >= window:
                break
        return rows[-window:]

    def observe(self, inputs: list[dict], prices: list[float]) -> None:
        now = datetime.now(timezone.utc)
        rows = [{**row, PREDICTION: price, "model_version": self.model_version,
                 "timestamp": now.isoformat(timespec="seconds")}
                for row, price in zip(inputs, prices)]
        self.recent.extend(rows)
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            with open(self.log_dir / f"predictions-{now:%Y-%m}.jsonl", "a") as f:
                f.writelines(json.dumps(row) + "\n" for row in rows)
        except OSError as e:
            # Sin registro se pierde historia, no la predicción.
            logger.warning(f"Could not write the prediction log: {e}")
        self._publish()

    def _publish(self) -> None:
        scores = drift_scores(self.profile, list(self.recent)) if self.profile and self.recent else {}
        metrics.publish_drift(len(self.recent), scores)
