#!/bin/sh
# Ciclo de entrenamiento completo: validación -> datos -> modelo registrado en MLflow
# (@challenger) -> puerta de calidad -> champion vs challenger -> @champion.
# Necesita MLFLOW_TRACKING_URI. Con PROMOTE=false solo registra el challenger.
# DATASET: CSV de King County con el que entrenar (por defecto el archivo
# completo; en el despliegue, el dataset vivo de scripts/release_sales.py).
# Salida 3: el challenger perdió contra el champion (no es un error del ciclo).
# Salida 5: el dataset no cumple el contrato (src/data/validate_data.py).
set -eu

DATASET="${DATASET:-data/raw/kc_house_data.csv}"
python src/data/validate_data.py --input "$DATASET"
python src/data/run_processing.py --input "$DATASET"
python src/models/train_model.py \
  --config configs/model_config.yaml \
  --data data/processed/cleaned_house_data.csv \
  --raw-data "$DATASET" \
  --models-dir models
if [ "${PROMOTE:-true}" = "true" ]; then
  python scripts/promote_model.py --config configs/model_config.yaml
fi
