#!/bin/sh
# Restauración del registro de modelos desde los backups de backup-registry.sh.
#
#   restore-registry.sh --check [DUMP]   prueba: restaura en un Postgres temporal y
#                                        comprueba que cada versión del modelo tiene
#                                        sus archivos en el espejo. No toca producción.
#   restore-registry.sh --yes [DUMP]     restauración real: reemplaza la base de MLflow
#                                        y repone en el bucket los artefactos que falten.
#
# DUMP: por defecto el más reciente de BACKUP_DIR/db. Antes de la restauración
# real se guarda la base actual en BACKUP_DIR/db/pre-restore-*.dump (no la
# borra la retención), para poder deshacerla con este mismo script.
set -eu

DEPLOY=/home/ubuntu/house-price-predictor
BACKUP_DIR="${BACKUP_DIR:-/home/ubuntu/backups/house-price}"
ARTIFACTS="$BACKUP_DIR/artifacts"
MC_IMAGE=minio/mc:RELEASE.2024-11-21T17-21-54Z
PG_IMAGE=postgres:16-alpine
BUCKET=house-price-mlflow
DB=house-price-mlflow-db

usage() {
    sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'
    exit 2
}

[ $# -ge 1 ] || usage
MODE=$1
case "$MODE" in --check|--yes) ;; *) usage ;; esac
DUMP="${2:-$(find "$BACKUP_DIR/db" -maxdepth 1 -name 'mlflow-*.dump' | sort | tail -1)}"
[ -n "$DUMP" ] && [ -f "$DUMP" ] || { echo "No dump found (${DUMP:-$BACKUP_DIR/db/mlflow-*.dump})" >&2; exit 1; }
echo "Dump: $DUMP"

# psql sobre el contenedor $1, salida sin formato.
query() {
    docker exec "$1" psql -U mlflow -d mlflow -tAF ' ' -c "$2"
}

# Cada versión registrada tiene que tener su modelo (MLmodel) en el espejo, y
# el run del @champion los archivos que la API carga junto al modelo.
check_artifacts() {
    container=$1
    failures=$(mktemp)
    versions=$(query "$container" "select version, storage_location from model_versions order by version")
    echo "$versions" | while read -r version location; do
        path="$ARTIFACTS/${location#mlflow-artifacts:/}"
        if [ -f "$path/MLmodel" ]; then
            echo "  v$version: model OK"
        else
            echo "  v$version: MISSING $path/MLmodel"
            echo x >>"$failures"
        fi
    done
    champion=$(query "$container" "select r.artifact_uri from runs r
        join model_versions v on v.run_id = r.run_uuid
        join registered_model_aliases a on a.name = v.name and a.version = v.version
        where a.alias = 'champion'")
    serving="$ARTIFACTS/${champion#mlflow-artifacts:/}/serving/model_metadata.json"
    if [ -n "$champion" ] && [ -f "$serving" ]; then
        echo "  @champion: serving files OK"
    else
        echo "  @champion: MISSING ${serving}"
        echo x >>"$failures"
    fi
    # El while corre en una subshell (tubería): los fallos se cuentan en un archivo.
    count=$(wc -l <"$failures")
    rm -f "$failures"
    if [ "$count" -gt 0 ]; then
        echo "$count checks failed" >&2
        return 1
    fi
}

if [ "$MODE" = "--check" ]; then
    container="hp-restore-check-$$"
    # El cron la corre cada mes: si falla, avisa como un backup fallido.
    finish() {
        rc=$?
        docker rm -f "$container" >/dev/null 2>&1 || true
        [ "$rc" -eq 0 ] && return
        cd "$DEPLOY" && NOTIFY_DETAIL="prueba de restauración de $(basename "$DUMP"), exit $rc" \
            docker compose run --rm --no-deps -T -e NOTIFY_DETAIL trainer python scripts/notify.py backup_failed \
            >/dev/null 2>&1 || echo "Could not send the failure notification" >&2
    }
    trap finish EXIT
    docker run -d --rm --name "$container" -e POSTGRES_USER=mlflow -e POSTGRES_PASSWORD=check \
        -e POSTGRES_DB=mlflow "$PG_IMAGE" >/dev/null
    # initdb arranca el servidor dos veces: se espera a que responda una consulta.
    for _ in $(seq 1 30); do
        docker exec "$container" psql -U mlflow -d mlflow -c 'select 1' >/dev/null 2>&1 && break
        sleep 2
    done
    docker exec -i "$container" pg_restore -U mlflow -d mlflow --no-owner --exit-on-error <"$DUMP"
    echo "Restored into a temporary Postgres:"
    echo "  $(query "$container" "select count(*) from model_versions") model versions," \
         "$(query "$container" "select count(*) from runs") runs," \
         "aliases: $(query "$container" "select string_agg(alias || '=v' || version, ' ' order by alias) from registered_model_aliases")"
    check_artifacts "$container"
    echo "Restore check OK"
    exit 0
fi

# ── Restauración real ──────────────────────────────────────────────────────
STAMP=$(date +%Y%m%d-%H%M%S)
safety="$BACKUP_DIR/db/pre-restore-$STAMP.dump"
docker exec "$DB" pg_dump -U mlflow -d mlflow --format=custom >"$safety"
echo "Current database saved to $safety"

cd "$DEPLOY"
docker compose stop mlflow
docker exec "$DB" psql -U mlflow -d postgres -v ON_ERROR_STOP=1 \
    -c "drop database mlflow with (force)" -c "create database mlflow owner mlflow"
docker exec -i "$DB" pg_restore -U mlflow -d mlflow --no-owner --exit-on-error <"$DUMP"
echo "Database restored from $DUMP"

# Sin --overwrite: solo se suben los objetos que faltan en el bucket.
docker run --rm --network minio-net --user "$(id -u):$(id -g)" \
    -e MC_CONFIG_DIR=/tmp/mc \
    -e "MC_HOST_registro=http://$(grep '^MLFLOW_S3_ACCESS_KEY=' .env | cut -d= -f2-):$(grep '^MLFLOW_S3_SECRET_KEY=' .env | cut -d= -f2-)@minio:9000" \
    -v "$ARTIFACTS:/backup:ro" \
    "$MC_IMAGE" mirror --quiet /backup "registro/$BUCKET" >/dev/null
echo "Missing artifacts copied back to $BUCKET"
check_artifacts "$DB"

docker compose up -d --wait mlflow
docker compose restart house-price-api
echo "Restore OK: MLflow is back and the API reloaded @champion. Undo with: $0 --yes $safety"
