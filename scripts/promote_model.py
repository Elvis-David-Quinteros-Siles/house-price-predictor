"""
Promueve una versión del registro de MLflow a @champion, la que sirve la API.

Uso:
  python scripts/promote_model.py                       # @challenger, si cumple la puerta de calidad
  python scripts/promote_model.py --version 3           # una versión concreta (rollback)
  python scripts/promote_model.py --version 3 --skip-gate

La API carga @champion al arrancar: tras promover, reiníciala.
Necesita MLFLOW_TRACKING_URI.
"""
import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml
from mlflow.tracking import MlflowClient

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_quality_gate import load_gate, passes  # noqa: E402

CHAMPION, CHALLENGER = "champion", "challenger"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/model_config.yaml")
    parser.add_argument("--version", help=f"Version to promote (default: @{CHALLENGER})")
    parser.add_argument("--skip-gate", action="store_true", help="Promote even if it fails the quality gate")
    return parser.parse_args()


def main(args):
    with open(args.config) as f:
        name = yaml.safe_load(f)["model"]["name"]
    client = MlflowClient()

    candidate = (client.get_model_version(name, args.version) if args.version
                 else client.get_model_version_by_alias(name, CHALLENGER))
    metrics = client.get_run(candidate.run_id).data.metrics
    print(f"{name} version {candidate.version} (run {candidate.run_id})")

    if not passes(metrics, load_gate(args.config)):
        if not args.skip_gate:
            sys.exit(f"Quality gate failed: version {candidate.version} stays out of production.")
        print("Quality gate failed, promoting anyway (--skip-gate).")

    try:
        previous = client.get_model_version_by_alias(name, CHAMPION).version
    except Exception:
        previous = None
    client.set_registered_model_alias(name, CHAMPION, candidate.version)
    client.set_model_version_tag(name, candidate.version, "promoted_at",
                                 datetime.now(timezone.utc).isoformat(timespec="seconds"))
    print(f"@{CHAMPION}: version {previous or '-'} -> {candidate.version}")


if __name__ == "__main__":
    main(parse_args())
