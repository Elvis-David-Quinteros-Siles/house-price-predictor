"""
Promueve una versión del registro de MLflow a @champion, la que sirve la API.

Uso:
  python scripts/promote_model.py                       # @challenger, si cumple la puerta de calidad
                                                        # y no pierde contra el @champion actual
  python scripts/promote_model.py --version 3           # una versión concreta (rollback, sin comparar)
  python scripts/promote_model.py --version 3 --skip-gate
  python scripts/promote_model.py --skip-comparison     # @challenger aunque pierda contra el champion

La API carga @champion al arrancar: tras promover, reiníciala.
Necesita MLFLOW_TRACKING_URI.
"""
import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import yaml
from mlflow.tracking import MlflowClient

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_quality_gate import load_gate, passes  # noqa: E402
from compare_models import challenger_wins, compare, load_rule  # noqa: E402

CHAMPION, CHALLENGER = "champion", "challenger"
# Código de salida cuando el challenger pierde: el ciclo funcionó, solo que el
# modelo nuevo no es mejor (el job de Jenkins lo marca como inestable, no fallido).
CHALLENGER_LOST = 3


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/model_config.yaml")
    parser.add_argument("--version", help=f"Version to promote (default: @{CHALLENGER})")
    parser.add_argument("--skip-gate", action="store_true", help="Promote even if it fails the quality gate")
    parser.add_argument("--skip-comparison", action="store_true",
                        help=f"Promote @{CHALLENGER} even if it loses against @{CHAMPION}")
    return parser.parse_args()


def champion_vs_challenger(client, name, champion, challenger, config_path):
    """Enfrenta las dos versiones en el mismo test y deja el resultado en MLflow."""
    print(f"Champion v{champion} vs challenger v{challenger}")
    metrics = compare(name, champion, challenger)
    wins = challenger_wins(metrics[CHAMPION], metrics[CHALLENGER], load_rule(config_path))

    # Un run por enfrentamiento, en el experimento del modelo: el historial de
    # decisiones queda en la UI de MLflow.
    mlflow.set_experiment(name)
    with mlflow.start_run(run_name=f"v{champion} vs v{challenger}"):
        mlflow.log_metrics({f"{role}_{k}": v for role, m in metrics.items() for k, v in m.items()})
        mlflow.set_tags({"stage": "champion_vs_challenger", "champion_version": champion,
                         "challenger_version": challenger, "challenger_wins": str(wins).lower()})
    client.set_model_version_tag(
        name, challenger, "vs_champion",
        f"v{champion}: MAE {metrics[CHAMPION]['mae']:,.0f} -> {metrics[CHALLENGER]['mae']:,.0f} "
        f"({'wins' if wins else 'loses'})",
    )
    return wins


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

    # Solo el challenger compite: un rollback (--version) es una decisión explícita.
    if not args.version and previous and previous != candidate.version:
        if not champion_vs_challenger(client, name, previous, candidate.version, args.config):
            if not args.skip_comparison:
                print(f"Challenger lost: version {candidate.version} stays @{CHALLENGER}, "
                      f"@{CHAMPION} remains version {previous}.", file=sys.stderr)
                sys.exit(CHALLENGER_LOST)
            print("Challenger lost, promoting anyway (--skip-comparison).")

    client.set_registered_model_alias(name, CHAMPION, candidate.version)
    client.set_model_version_tag(name, candidate.version, "promoted_at",
                                 datetime.now(timezone.utc).isoformat(timespec="seconds"))
    print(f"@{CHAMPION}: version {previous or '-'} -> {candidate.version}")


if __name__ == "__main__":
    main(parse_args())
