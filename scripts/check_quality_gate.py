"""
Puerta de calidad: falla si el modelo entrenado no alcanza los mínimos de
configs/model_config.yaml (sección quality_gate).

Uso: python scripts/check_quality_gate.py <model_metadata.json> [configs/model_config.yaml]
"""
import json
import sys

import yaml


def load_gate(config_path="configs/model_config.yaml"):
    with open(config_path) as f:
        return yaml.safe_load(f)["quality_gate"]


def passes(metrics, gate):
    """Imprime cada comprobación y devuelve si el modelo las cumple todas."""
    checks = [
        ("R²", metrics["r2"], ">=", gate["min_r2"], metrics["r2"] >= gate["min_r2"]),
        ("MAE", metrics["mae"], "<=", gate["max_mae"], metrics["mae"] <= gate["max_mae"]),
        ("MAPE", metrics["mape"], "<=", gate["max_mape"], metrics["mape"] <= gate["max_mape"]),
    ]
    for name, value, op, limit, ok in checks:
        print(f"{'OK  ' if ok else 'FAIL'} {name}: {value:,.4f} (required {op} {limit:,})")
    return all(ok for *_, ok in checks)


def main(metadata_path, config_path="configs/model_config.yaml"):
    with open(metadata_path) as f:
        metrics = json.load(f)["metrics"]
    if not passes(metrics, load_gate(config_path)):
        sys.exit("Quality gate failed: the new model is worse than the required minimum.")
    print("Quality gate passed.")


if __name__ == "__main__":
    main(*sys.argv[1:])
