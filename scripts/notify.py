"""
Aviso por correo de lo que decidió el entrenamiento continuo (job house-price-ct).

Uso:
  python scripts/notify.py promoted --detail "v5 en producción" --build-url URL
  python scripts/notify.py failed --detail "Entrenamiento o puerta de calidad: exit 1"

Eventos: promoted, challenger_lost, degraded_lost, drift_no_data, invalid_data, failed,
backup_failed (deployment/backup/backup-registry.sh) y test (comprueba la
configuración SMTP).
El correo lleva el estado del registro (champion, challenger, error real) si
MLflow responde; si no, sale igual, sin ese resumen.

SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD y NOTIFY_TO vienen del .env del
despliegue. Sin SMTP_HOST no se envía nada (desarrollo local, CI). Salida 1
si el envío falla: el job lo anota y sigue, un aviso perdido no tumba el ciclo.
"""
import argparse
import os
import smtplib
import sys
from email.message import EmailMessage

import yaml

# (asunto, qué significa y qué hacer)
EVENTS = {
    "promoted": (
        "Nuevo modelo en producción",
        "El challenger pasó la puerta de calidad, ganó al champion en el mismo test y la API ya lo sirve.",
    ),
    "challenger_lost": (
        "El challenger perdió",
        "El modelo nuevo no mejora al champion: sigue en producción la versión actual. "
        "La comparación quedó en MLflow (run 'vN vs vM').",
    ),
    "degraded_lost": (
        "Champion degradado y el challenger no lo supera",
        "El modelo en producción ya falla con las ventas reales y reentrenar con los datos nuevos no lo "
        "arregla. Revisar Grafana (House Price Predictor -> Error real) y MLflow; probar otro "
        "entrenamiento o un rollback (scripts/promote_model.py --version N).",
    ),
    "drift_no_data": (
        "Drift sin ventas nuevas",
        "La alerta DriftDeDatosDelModelo está activa pero no hay ventas nuevas con precio: reentrenar daría "
        "el mismo modelo. Revisar qué casas se tasan (Grafana -> House Price Predictor -> Drift de datos).",
    ),
    "invalid_data": (
        "Datos inválidos: no se entrenó",
        "El dataset vivo no cumple el contrato (src/data/validate_data.py): el ciclo paró antes de medir "
        "el error real y de entrenar, y sigue el champion. Corregir los datos (o reiniciar la simulación "
        "con INIT_UNTIL) y volver a lanzar house-price-ct.",
    ),
    "failed": (
        "Falló el entrenamiento continuo",
        "El ciclo se cortó con un error: el champion no cambió. Ver el log del build.",
    ),
    "backup_failed": (
        "Falló el backup del registro de modelos",
        "El backup diario (deployment/backup/backup-registry.sh) no terminó: el último bueno es el de "
        "~/backups/house-price/last-success. Ver ~/backups/house-price/backup.log.",
    ),
    "test": (
        "Prueba de avisos",
        "Correo de prueba: la configuración SMTP del entrenamiento continuo funciona.",
    ),
}
PREFIX = "[house-price]"


def registry_summary(config_path="configs/model_config.yaml") -> str:
    """Champion, challenger y último error real según MLflow, o '' si no responde."""
    # Un aviso de fallo no puede quedarse colgado esperando a un MLflow caído.
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "0")
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "10")
    try:
        from mlflow.tracking import MlflowClient

        with open(config_path) as f:
            name = yaml.safe_load(f)["model"]["name"]
        client = MlflowClient()
        lines, seen = [], set()
        for alias in ("champion", "challenger"):
            try:
                version = client.get_model_version_by_alias(name, alias)
            except Exception:
                continue
            if version.version in seen:
                lines.append(f"@{alias}: v{version.version} (el mismo)")
                continue
            seen.add(version.version)
            lines.append(f"@{alias}: v{version.version} - {version.description or ''}".rstrip(" -"))
            if "vs_champion" in version.tags:
                lines.append(f"  vs champion {version.tags['vs_champion']}")
            for tag in sorted(t for t in version.tags if t.startswith("real_error_"))[-1:]:
                lines.append(f"  error real con las ventas de {tag.removeprefix('real_error_')}: {version.tags[tag]}")
        return "\n".join(lines)
    except Exception as e:
        return f"(MLflow no respondió: {e})"


def compose(event: str, detail: str = "", build_url: str = "", summary: str = "") -> tuple[str, str]:
    """Asunto y cuerpo del correo."""
    title, meaning = EVENTS[event]
    subject = f"{PREFIX} {title}" + (f": {detail}" if detail else "")
    parts = [meaning]
    if detail:
        parts.append(f"Detalle: {detail}")
    if summary:
        parts.append(f"Registro de modelos:\n{summary}")
    if build_url:
        parts.append(f"Build: {build_url}")
    return subject, "\n\n".join(parts) + "\n"


def send(subject: str, body: str) -> None:
    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = f"House Price Predictor <{os.environ['SMTP_USER']}>"
    message["To"] = os.environ["NOTIFY_TO"]
    message.set_content(body)
    with smtplib.SMTP(os.environ["SMTP_HOST"], int(os.getenv("SMTP_PORT", "587")), timeout=30) as smtp:
        smtp.starttls()
        smtp.login(os.environ["SMTP_USER"], os.environ["SMTP_PASSWORD"])
        smtp.send_message(message)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("event", choices=sorted(EVENTS))
    parser.add_argument("--detail", default=os.getenv("NOTIFY_DETAIL", ""))
    parser.add_argument("--build-url", default=os.getenv("BUILD_URL", ""))
    parser.add_argument("--config", default="configs/model_config.yaml")
    return parser.parse_args()


def main(args):
    subject, body = compose(args.event, args.detail, args.build_url, registry_summary(args.config))
    print(f"{subject}\n\n{body}")
    if not os.getenv("SMTP_HOST"):
        print("SMTP_HOST not set: not sent.")
        return 0
    try:
        send(subject, body)
    except Exception as e:
        print(f"Could not send the notification: {e}", file=sys.stderr)
        return 1
    print(f"Sent to {os.environ['NOTIFY_TO']}")
    return 0


if __name__ == "__main__":
    sys.exit(main(parse_args()))
