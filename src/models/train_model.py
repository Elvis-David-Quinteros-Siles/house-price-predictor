import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
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
from engineer import INPUT_FEATURES, build_pipeline  # noqa: E402

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
    parser.add_argument("--models-dir", type=str, required=True, help="Directory to save trained model")
    parser.add_argument("--mlflow-tracking-uri", type=str, default=None, help="MLflow tracking URI")
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

    if args.mlflow_tracking_uri:
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

        # Log and register model
        mlflow.sklearn.log_model(model, "tuned_model")
        model_name = model_cfg['name']
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/tuned_model"

        logger.info("Registering model to MLflow Model Registry...")
        client = MlflowClient()
        try:
            client.create_registered_model(model_name)
        except mlflow.exceptions.RestException:
            pass  # already exists

        model_version = client.create_model_version(
            name=model_name,
            source=model_uri,
            run_id=mlflow.active_run().info.run_id
        )

        # Transition model to "Staging"
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Staging"
        )

        # Add a human-readable description
        description = (
            f"Model for predicting house prices.\n"
            f"Algorithm: {model_cfg['best_model']}\n"
            f"Hyperparameters: {model_cfg['parameters']}\n"
            f"Features used: {', '.join(INPUT_FEATURES)} + sale_month + neighbour prices\n"
            f"Target variable: {target}\n"
            f"Trained on dataset: {args.data}\n"
            f"Model saved at: {args.models_dir}/trained/{model_name}.pkl\n"
            f"Performance metrics:\n"
            f"  - MAE: {mae:.2f}\n"
            f"  - R²: {r2:.4f}"
        )
        client.update_registered_model(name=model_name, description=description)

        # Add tags for better organization
        client.set_registered_model_tag(model_name, "algorithm", model_cfg['best_model'])
        client.set_registered_model_tag(model_name, "hyperparameters", str(model_cfg['parameters']))
        client.set_registered_model_tag(model_name, "features", ",".join(INPUT_FEATURES))
        client.set_registered_model_tag(model_name, "target_variable", target)
        client.set_registered_model_tag(model_name, "training_dataset", args.data)
        client.set_registered_model_tag(model_name, "model_path", f"{args.models_dir}/trained/{model_name}.pkl")

        # Add dependency tags
        deps = {
            "python_version": platform.python_version(),
            "scikit_learn_version": sklearn.__version__,
            "lightgbm_version": lgb.__version__,
            "xgboost_version": xgb.__version__,
            "pandas_version": pd.__version__,
            "numpy_version": np.__version__,
        }
        for k, v in deps.items():
            client.set_registered_model_tag(model_name, k, v)

        # Save model locally
        save_path = f"{args.models_dir}/trained/{model_name}.pkl"
        joblib.dump(model, save_path)
        logger.info(f"Saved trained model to: {save_path}")

        # Coordenadas de las casas vendidas: la API rechaza ubicaciones lejos de
        # cualquier venta (agua, fuera del condado), donde el modelo no sabe nada.
        locations_path = f"{args.models_dir}/trained/sold_locations.npy"
        np.save(locations_path, data[['lat', 'long']].to_numpy())
        logger.info(f"Saved {len(data)} sold locations to {locations_path}")

        # Metadatos que la API devuelve junto a cada predicción.
        metadata = {
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
        metadata_path = f"{args.models_dir}/trained/model_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved model metadata to: {metadata_path}")
        logger.info(f"CV   MAE: {cv_metrics['mae']:.2f}, R²: {cv_metrics['r2']:.4f}, MAPE: {cv_metrics['mape']:.2%}")
        logger.info(f"Test MAE: {mae:.2f}, R²: {r2:.4f}, MAPE: {mape:.2%}")
        logger.info(f"90% interval ratio: [{interval_low:.3f}, {interval_high:.3f}], test coverage {coverage:.1%}")

if __name__ == "__main__":
    args = parse_args()
    main(args)
