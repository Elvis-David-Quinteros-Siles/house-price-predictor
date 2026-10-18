import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import hashlib
import os
from datetime import datetime, timezone
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.model_selection import KFold, cross_val_predict, train_test_split
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.compose import TransformedTargetRegressor
import lightgbm as lgb
import xgboost as xgb
import yaml
import json
import logging
from mlflow.tracking import MlflowClient
import platform
import sklearn

# El pipeline de variables vive en src/features/engineer.py (módulo `engineer`).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "features"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "monitoring"))
from engineer import INPUT_FEATURES, build_pipeline  # noqa: E402
from drift import build_profile  # noqa: E402

ENGINEER_PATH = Path(__file__).resolve().parents[1] / "features" / "engineer.py"
# Carpeta de artefactos del run con los ficheros que la API carga junto al modelo.
SERVING_ARTIFACTS = "serving"
# Commit del código entrenado; lo pasa el build (en la imagen no hay .git).
GIT_SHA = os.getenv("GIT_SHA", "unknown")


def file_sha256(path):
    """Huella del dataset, para saber con qué datos se entrenó cada versión."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()

# -----------------------------
# Configure logging
# -----------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# -----------------------------
# Argument parser
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Train and register final model from config.")
    parser.add_argument("--config", type=str, required=True, help="Path to model_config.yaml")
    parser.add_argument("--data", type=str, required=True, help="Path to the cleaned CSV (run_processing.py)")
    parser.add_argument("--raw-data", type=str, default="data/raw/kc_house_data.csv",
                        help="Raw dataset, whose hash is recorded with the model")
    parser.add_argument("--models-dir", type=str, required=True, help="Directory to save trained model")
    parser.add_argument("--mlflow-tracking-uri", type=str, default=os.getenv("MLFLOW_TRACKING_URI"),
                        help="MLflow tracking URI (default: $MLFLOW_TRACKING_URI)")
    return parser.parse_args()

# -----------------------------
# Load model from config
# -----------------------------
def get_model_instance(name, params):
    model_map = {
        'LinearRegression': LinearRegression,
        'RandomForest': RandomForestRegressor,
        'GradientBoosting': GradientBoostingRegressor,
        'LightGBM': lgb.LGBMRegressor,
        'XGBoost': xgb.XGBRegressor
    }
    if name not in model_map:
        raise ValueError(f"Unsupported model: {name}")
    return model_map[name](**params)

# -----------------------------
# Main logic
# -----------------------------
def main(args):
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    model_cfg = config['model']

    if not args.mlflow_tracking_uri:
        raise SystemExit("Set --mlflow-tracking-uri or MLFLOW_TRACKING_URI: every trained model goes to the registry")
    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment(model_cfg['name'])

    # Load data
    data = pd.read_csv(args.data)
    target = model_cfg['target_variable']
    X = data[INPUT_FEATURES + ['sale_month']]
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Pipeline completo (variables + preprocesado + modelo), ajustado solo con train.
    # Las variables de vecinos esperan el precio en log.
    if not model_cfg.get('log_target'):
        raise ValueError("The pipeline needs log_target: true (neighbour features use log prices)")
    model = TransformedTargetRegressor(
        regressor=build_pipeline(
            get_model_instance(model_cfg['best_model'], model_cfg['parameters']),
            neighbours=model_cfg.get('neighbours', 10),
        ),
        func=np.log, inverse_func=np.exp,
    )

    # Start MLflow run
    with mlflow.start_run(run_name="final_training"):
        # Validación cruzada de 5 folds dentro de train: métricas más estables que
        # un único split y residuos fuera de muestra para el intervalo.
        logger.info("Cross-validating on the training set")
        y_oof = cross_val_predict(model, X_train, y_train, cv=KFold(5, shuffle=True, random_state=42))
        cv_metrics = {
            'mae': float(mean_absolute_error(y_train, y_oof)),
            'r2': float(r2_score(y_train, y_oof)),
            'mape': float(mean_absolute_percentage_error(y_train, y_oof)),
        }

        # Intervalo empírico del 90%: cuantiles del cociente real/predicho en las
        # predicciones fuera de muestra de la CV. Se aplica como
        # [pred * low, pred * high] en inferencia y se comprueba en test.
        ratio = y_train.to_numpy() / y_oof
        interval_low, interval_high = (float(q) for q in np.quantile(ratio, [0.05, 0.95]))

        logger.info(f"Training model: {model_cfg['best_model']}")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mae = float(mean_absolute_error(y_test, y_pred))
        r2 = float(r2_score(y_test, y_pred))
        mape = float(mean_absolute_percentage_error(y_test, y_pred))
        test_ratio = y_test.to_numpy() / y_pred
        coverage = float(np.mean((test_ratio >= interval_low) & (test_ratio <= interval_high)))

        # Log params and metrics
        mlflow.log_params(model_cfg['parameters'])
        mlflow.log_metrics({'mae': mae, 'r2': r2, 'mape': mape, 'interval_coverage': coverage,
                            **{f'cv_{k}': v for k, v in cv_metrics.items()}})
        mlflow.set_tags({'algorithm': model_cfg['best_model'], 'git_sha': GIT_SHA,
                         'dataset_sha256': file_sha256(args.raw_data)})

        # Modelo completo (variables + regresor) al registro. engineer.py viaja con
        # el modelo porque el pickle referencia sus clases. La firma son las
        # entradas de la API: sale_month lo rellena el pipeline. Formato cloudpickle:
        # skops (el de serie en MLflow 3) exige declarar como confiables las clases
        # propias de engineer.py y las de LightGBM.
        model_name = model_cfg['name']
        example = X_test[INPUT_FEATURES].head(5)
        logger.info("Registering model to MLflow Model Registry...")
        logged = mlflow.sklearn.log_model(
            model,
            name="model",
            registered_model_name=model_name,
            signature=infer_signature(example, model.predict(example)),
            code_paths=[str(ENGINEER_PATH)],
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        )
        version = str(logged.registered_model_version)

        client = MlflowClient()
        client.update_model_version(
            name=model_name, version=version,
            description=(f"{model_cfg['best_model']} sobre log(precio), {len(X_train)} ventas de entrenamiento. "
                         f"Test: MAE ${mae:,.0f}, R² {r2:.4f}, MAPE {mape:.2%}."),
        )
        deps = {
            "python_version": platform.python_version(),
            "scikit_learn_version": sklearn.__version__,
            "lightgbm_version": lgb.__version__,
            "pandas_version": pd.__version__,
            "numpy_version": np.__version__,
        }
        for k, v in {'git_sha': GIT_SHA, 'dataset_sha256': file_sha256(args.raw_data), **deps}.items():
            client.set_model_version_tag(model_name, version, k, v)
        # Candidato a producción: scripts/promote_model.py lo pasa a champion si
        # cumple la puerta de calidad.
        client.set_registered_model_alias(model_name, "challenger", version)
        logger.info(f"Registered {model_name} version {version} as @challenger")

        # Save model locally
        trained_dir = Path(args.models_dir) / "trained"
        save_path = trained_dir / f"{model_name}.pkl"
        joblib.dump(model, save_path)
        logger.info(f"Saved trained model to: {save_path}")

        # Coordenadas de las casas vendidas: la API rechaza ubicaciones lejos de
        # cualquier venta (agua, fuera del condado), donde el modelo no sabe nada.
        locations_path = trained_dir / "sold_locations.npy"
        np.save(locations_path, data[['lat', 'long']].to_numpy())
        logger.info(f"Saved {len(data)} sold locations to {locations_path}")

        # Metadatos que la API devuelve junto a cada predicción.
        metadata = {
            "model_name": model_name,
            "model_version": version,
            "run_id": mlflow.active_run().info.run_id,
            "git_sha": GIT_SHA,
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "algorithm": model_cfg['best_model'],
            "log_target": bool(model_cfg.get('log_target', False)),
            "dataset": model_cfg.get('dataset', args.data),
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "metrics": {"mae": round(mae, 2), "r2": round(r2, 4), "mape": round(mape, 4)},
            "cv_metrics": {k: round(v, 4 if k != 'mae' else 2) for k, v in cv_metrics.items()},
            "interval": {"coverage": 0.9, "low_ratio": round(interval_low, 4), "high_ratio": round(interval_high, 4),
                         "test_coverage": round(coverage, 4)},
        }
        metadata_path = trained_dir / "model_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved model metadata to: {metadata_path}")

        # Referencia para el drift: entradas de entrenamiento y sus predicciones
        # fuera de muestra (las de producción también lo son).
        profile_path = trained_dir / "reference_profile.json"
        with open(profile_path, "w") as f:
            json.dump(build_profile(X_train, y_oof), f)
        logger.info(f"Saved drift reference profile to: {profile_path}")

        # Lo que la API necesita además del modelo, junto a la versión registrada.
        mlflow.log_artifact(str(metadata_path), SERVING_ARTIFACTS)
        mlflow.log_artifact(str(locations_path), SERVING_ARTIFACTS)
        mlflow.log_artifact(str(profile_path), SERVING_ARTIFACTS)

        logger.info(f"CV   MAE: {cv_metrics['mae']:.2f}, R²: {cv_metrics['r2']:.4f}, MAPE: {cv_metrics['mape']:.2%}")
        logger.info(f"Test MAE: {mae:.2f}, R²: {r2:.4f}, MAPE: {mape:.2%}")
        logger.info(f"90% interval ratio: [{interval_low:.3f}, {interval_high:.3f}], test coverage {coverage:.1%}")

if __name__ == "__main__":
    args = parse_args()
    main(args)
