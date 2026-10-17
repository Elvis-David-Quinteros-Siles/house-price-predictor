#!/bin/sh
# Ciclo de entrenamiento completo: datos -> modelo registrado en MLflow
# (@challenger) -> puerta de calidad -> @champion.
# Necesita MLFLOW_TRACKING_URI. Con PROMOTE=false solo registra el challenger.
set -eu

python src/data/run_processing.py
python src/models/train_model.py \
  --config configs/model_config.yaml \
  --data data/processed/cleaned_house_data.csv \
  --models-dir models
if [ "${PROMOTE:-true}" = "true" ]; then
  python scripts/promote_model.py --config configs/model_config.yaml
fi
