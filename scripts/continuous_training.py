"""
Comprobaciones del entrenamiento continuo (deployment/jenkins/continuous-training.groovy).

  python scripts/continuous_training.py new-data [--dataset PATH]
      exit 0 si el dataset no es con el que se entrenó @champion (hay que
      reentrenar), 10 si es el mismo.
  python scripts/continuous_training.py drift [--prometheus URL]
      exit 0 si la alerta DriftDeDatosDelModelo está activa, 10 si no.
  python scripts/continuous_training.py champion
      imprime la versión @champion.

new-data y champion necesitan MLFLOW_TRACKING_URI.
"""
import argparse
import hashlib
import json
import os
import sys
import urllib.parse
import urllib.request

import yaml

YES, NO = 0, 10
DRIFT_ALERT = "DriftDeDatosDelModelo"


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_name(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)["model"]["name"]


def champion(config_path):
    from mlflow.tracking import MlflowClient
    return MlflowClient().get_model_version_by_alias(model_name(config_path), "champion")


def new_data(args):
    current = champion(args.config)
    trained_on = current.tags.get("dataset_sha256")
    dataset = file_sha256(args.dataset)
    print(f"@champion v{current.version}: data until {current.tags.get('data_until', '?')}, "
          f"dataset {str(trained_on)[:12]}")
    print(f"{args.dataset}: {dataset[:12]}")
    if dataset == trained_on:
        print("Same dataset: nothing new to learn.")
        return NO
    print("New data: retrain.")
    return YES


def drift(args):
    query = urllib.parse.urlencode({"query": f'ALERTS{{alertname="{DRIFT_ALERT}",alertstate="firing"}}'})
    with urllib.request.urlopen(f"{args.prometheus}/api/v1/query?{query}", timeout=10) as response:
        firing = json.load(response)["data"]["result"]
    for alert in firing:
        print(f"{DRIFT_ALERT} firing: {alert['metric'].get('feature')}")
    if not firing:
        print(f"{DRIFT_ALERT} not firing.")
    return YES if firing else NO


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/model_config.yaml")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("new-data").add_argument(
        "--dataset", default=os.getenv("DATASET", "data/raw/kc_house_data.csv"))
    sub.add_parser("drift").add_argument("--prometheus", default="http://core-prometheus:9090")
    sub.add_parser("champion")
    return parser.parse_args()


def main(args):
    if args.command == "champion":
        print(champion(args.config).version)
        return 0
    return {"new-data": new_data, "drift": drift}[args.command](args)


if __name__ == "__main__":
    sys.exit(main(parse_args()))
