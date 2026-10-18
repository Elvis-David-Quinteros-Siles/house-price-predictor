"""
Error real del @champion con las ventas que llegaron después de entrenarlo.

El job house-price-ct lo corre al llegar ventas nuevas, antes de reentrenar:
el champion tasa las casas de cada mes posterior a su tag data_until y se
compara con el precio de venta (src/monitoring/performance.py). Cada mes
queda como un run en MLflow y como tag de la versión; el último, además, en
--output, que la API publica en /metrics para Prometheus.

Uso:
  python scripts/production_performance.py                   # dataset vivo ($DATASET)
  python scripts/production_performance.py --dataset data/raw/kc_house_data.csv --dry-run

Salida 0 si el error está dentro del límite (o no hay ventas que el champion
no viera), 4 si el champion está degradado. Necesita MLFLOW_TRACKING_URI.
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from mlflow.tracking import MlflowClient

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src" / "data"), str(ROOT / "src" / "features"), str(ROOT / "src" / "monitoring")]
from engineer import INPUT_FEATURES  # noqa: E402
from performance import (REAL_ERROR_FILE, degradation, evaluate_sales, mape_limit, month_label,  # noqa: E402
                         unseen_months)
from run_processing import clean_data  # noqa: E402

CHAMPION = "champion"
# Código de salida con el champion degradado: el job de Jenkins lo anota y
# sigue con el reentrenamiento.
DEGRADED = 4
STAGE = "production_performance"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/model_config.yaml")
    parser.add_argument("--dataset", default=os.getenv("DATASET", "data/raw/kc_house_data.csv"),
                        help="King County CSV with the sales published so far")
    parser.add_argument("--output", default=f"data/performance/{REAL_ERROR_FILE}",
                        help="Latest evaluation, read by the API")
    parser.add_argument("--dry-run", action="store_true", help="Only print: nothing to MLflow or --output")
    return parser.parse_args()


def already_logged(client, experiment_id, version, month):
    runs = client.search_runs(
        [experiment_id],
        f"tags.stage = '{STAGE}' and tags.model_version = '{version}' and tags.month = '{month}'",
        max_results=1,
    )
    return bool(runs)


def log_month(client, name, version, month, result, limit, degraded):
    """Un run por versión y mes en el experimento del modelo, y el resumen como tag de la versión."""
    experiment = mlflow.set_experiment(name)
    if already_logged(client, experiment.experiment_id, version, month):
        print(f"  v{version} {month} already in MLflow")
        return
    with mlflow.start_run(run_name=f"v{version} con ventas de {month}"):
        mlflow.log_metrics({**result, "mape_limit": limit})
        mlflow.set_tags({"stage": STAGE, "model_version": version, "month": month,
                         "degraded": str(degraded).lower()})
    client.set_model_version_tag(name, version, f"real_error_{month}",
                                 f"MAPE {result['mape']:.2%}, sesgo {result['bias']:+.1%}, "
                                 f"{result['sales']} ventas")


def main(args):
    with open(args.config) as f:
        config = yaml.safe_load(f)
    name, rule = config["model"]["name"], config["production_performance"]
    client = MlflowClient()

    champion = client.get_model_version_by_alias(name, CHAMPION)
    data_until = champion.tags.get("data_until")
    if not data_until:
        sys.exit(f"@{CHAMPION} v{champion.version} has no data_until tag: cannot tell which sales are new")
    test_mape = client.get_run(champion.run_id).data.metrics["mape"]
    limit = mape_limit(test_mape, rule)

    sales = clean_data(pd.read_csv(args.dataset, dtype={"id": str, "zipcode": str}))
    months = unseen_months(sales["sale_month"], data_until)
    print(f"@{CHAMPION} v{champion.version}: data until {data_until}, test MAPE {test_mape:.2%}, "
          f"limit {limit:.2%}")
    if not months:
        print("No sales after the champion's training data: nothing to evaluate.")
        return 0

    metadata = json.loads(Path(mlflow.artifacts.download_artifacts(
        run_id=champion.run_id, artifact_path="serving/model_metadata.json")).read_text())
    model = mlflow.sklearn.load_model(f"models:/{name}/{champion.version}")

    print(f"{'month':8}{'sales':>7}{'MAE':>10}{'MAPE':>8}{'bias':>8}{'in 90%':>8}")
    for sale_month in months:
        month = month_label(sale_month)
        sold = sales[sales["sale_month"] == sale_month]
        # Sin sale_month, como tasa la API: con el último mes de entrenamiento.
        result = evaluate_sales(sold["price"], model.predict(sold[INPUT_FEATURES]), metadata["interval"])
        reasons = degradation(result, limit, rule)
        print(f"{month:8}{result['sales']:7}{result['mae']:10,.0f}{result['mape']:8.2%}{result['bias']:+8.1%}"
              f"{result['interval_coverage']:8.1%}  {'DEGRADED: ' + ', '.join(reasons) if reasons else 'ok'}")
        if not args.dry_run:
            log_month(client, name, champion.version, month, result, limit, bool(reasons))

    # Para Prometheus, el último mes: el que acaba de llegar.
    if not args.dry_run:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({
            "model_version": champion.version, "month": month, **result,
            "test_mape": test_mape, "mape_limit": limit, "max_abs_bias": rule["max_abs_bias"],
            "degraded": bool(reasons), "reasons": reasons,
            "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }, indent=2))
        print(f"Latest evaluation written to {output}")

    if reasons:
        print(f"@{CHAMPION} v{champion.version} degraded on {month} sales: {', '.join(reasons)}", file=sys.stderr)
        return DEGRADED
    return 0


if __name__ == "__main__":
    sys.exit(main(parse_args()))
