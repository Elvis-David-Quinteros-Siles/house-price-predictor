"""
Champion vs challenger: evalúa las dos versiones sobre el mismo test (el del
challenger, guardado con su run) y decide si el challenger reemplaza al champion.

Las métricas registradas de cada versión no son comparables si se entrenaron
con datos distintos: cada una se midió en su propio test. Aquí las dos predicen
exactamente las mismas casas.
"""
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score

# Test del run de entrenamiento: entradas del modelo y precio real (train_model.py).
HOLDOUT_ARTIFACT = "evaluation/holdout.csv"
TARGET = "price"


def load_rule(config_path="configs/model_config.yaml"):
    with open(config_path) as f:
        return yaml.safe_load(f)["champion_challenger"]


def evaluate(model, X, y):
    y_pred = model.predict(X)
    return {
        "mae": float(mean_absolute_error(y, y_pred)),
        "r2": float(r2_score(y, y_pred)),
        "mape": float(mean_absolute_percentage_error(y, y_pred)),
    }


def challenger_wins(champion, challenger, rule):
    """Imprime la comparación y devuelve si el challenger debe pasar a champion.

    Decide el MAE, la métrica con la que se eligió el modelo: el challenger gana
    si no es peor que el champion en más de max_mae_increase (relativo). El
    margen deja promover un modelo con datos más recientes que empata dentro
    del ruido.
    """
    limit = champion["mae"] * (1 + rule["max_mae_increase"])
    print(f"{'':6}{'champion':>12}{'challenger':>12}")
    for metric, fmt in (("mae", "{:12,.0f}"), ("r2", "{:12.4f}"), ("mape", "{:12.2%}")):
        print(f"{metric.upper():6}{fmt.format(champion[metric])}{fmt.format(challenger[metric])}")
    wins = challenger["mae"] <= limit
    print(f"{'OK  ' if wins else 'FAIL'} challenger MAE {challenger['mae']:,.0f} "
          f"(required <= {limit:,.0f}, champion + {rule['max_mae_increase']:.1%})")
    return wins


def compare(name, champion_version, challenger_version):
    """Métricas de las dos versiones sobre el test del challenger."""
    run_id = mlflow.MlflowClient().get_model_version(name, challenger_version).run_id
    holdout = pd.read_csv(mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=HOLDOUT_ARTIFACT))
    X, y = holdout.drop(columns=[TARGET]), holdout[TARGET]
    print(f"Holdout: {len(holdout)} sales from run {run_id}")
    return {
        role: evaluate(mlflow.sklearn.load_model(f"models:/{name}/{version}"), X, y)
        for role, version in (("champion", champion_version), ("challenger", challenger_version))
    }
