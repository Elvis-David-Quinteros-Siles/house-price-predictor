#!/bin/sh
# Backup del registro de modelos y de los datos de producción del House Price
# Predictor. Lo corre el cron del host una vez al día (README.md de esta carpeta).
#
#   BACKUP_DIR/db/mlflow-AAAAMMDD-HHMMSS.dump       pg_dump de MLflow (versiones, alias, runs, métricas)
#   BACKUP_DIR/artifacts/                           espejo acumulado del bucket house-price-mlflow
#   BACKUP_DIR/volumes/<volumen>-AAAAMMDD-HHMMSS.tar.gz   prediction-log y training-data
#
# Primero la base y después los artefactos: así el espejo siempre tiene los
# archivos de todo lo que referencia el dump más reciente. El espejo no borra
# nunca (los artefactos de MLflow no cambian una vez escritos): un objeto
# borrado por error en el bucket sigue aquí.
# Se guardan los últimos BACKUP_KEEP dumps y tarballs. Si falla, avisa por
# correo (scripts/notify.py backup_failed).
set -eu

DEPLOY=/home/ubuntu/house-price-predictor
BACKUP_DIR="${BACKUP_DIR:-/home/ubuntu/backups/house-price}"
KEEP="${BACKUP_KEEP:-14}"
MC_IMAGE=minio/mc:RELEASE.2024-11-21T17-21-54Z
# Imagen con tar y gzip que ya está en el servidor.
TAR_IMAGE=postgres:16-alpine
BUCKET=house-price-mlflow
VOLUMES="prediction-log training-data"
STAMP=$(date +%Y%m%d-%H%M%S)
STEP=inicio

notify_failure() {
    rc=$?
    [ "$rc" -eq 0 ] && return
    rm -f "$BACKUP_DIR"/db/*.tmp
    echo "Backup FAILED at step '$STEP' (exit $rc)" >&2
    cd "$DEPLOY" && NOTIFY_DETAIL="paso '$STEP', exit $rc" \
        docker compose run --rm --no-deps -T -e NOTIFY_DETAIL trainer python scripts/notify.py backup_failed \
        >/dev/null 2>&1 || echo "Could not send the failure notification" >&2
}
trap notify_failure EXIT

env_value() {
    grep "^$1=" "$DEPLOY/.env" | tail -1 | cut -d= -f2-
}

mkdir -p "$BACKUP_DIR/db" "$BACKUP_DIR/artifacts" "$BACKUP_DIR/volumes"

STEP=postgres
dump="$BACKUP_DIR/db/mlflow-$STAMP.dump"
docker exec house-price-mlflow-db pg_dump -U mlflow -d mlflow --format=custom >"$dump.tmp"
# Un dump que pg_restore no sabe leer no es un backup.
docker exec -i house-price-mlflow-db pg_restore --list <"$dump.tmp" >/dev/null
mv "$dump.tmp" "$dump"
echo "Postgres: $dump ($(du -h "$dump" | cut -f1))"

STEP=artefactos
# Las credenciales de MLflow para MinIO (usuario limitado al bucket) bastan para leerlo.
docker run --rm --network minio-net --user "$(id -u):$(id -g)" \
    -e MC_CONFIG_DIR=/tmp/mc \
    -e "MC_HOST_registro=http://$(env_value MLFLOW_S3_ACCESS_KEY):$(env_value MLFLOW_S3_SECRET_KEY)@minio:9000" \
    -v "$BACKUP_DIR/artifacts:/backup" \
    "$MC_IMAGE" mirror --quiet --overwrite "registro/$BUCKET" /backup >/dev/null
echo "Artifacts: $BACKUP_DIR/artifacts ($(find "$BACKUP_DIR/artifacts" -type f | wc -l) files, $(du -sh "$BACKUP_DIR/artifacts" | cut -f1))"

STEP=volumenes
for volume in $VOLUMES; do
    tarball="$BACKUP_DIR/volumes/$volume-$STAMP.tar.gz"
    docker run --rm --user "$(id -u):$(id -g)" \
        -v "house-price-predictor_$volume:/data:ro" -v "$BACKUP_DIR/volumes:/backup" \
        "$TAR_IMAGE" tar czf "/backup/$volume-$STAMP.tar.gz" -C /data .
    echo "Volume $volume: $tarball ($(du -h "$tarball" | cut -f1))"
done

STEP=retencion
prune() {
    # $1: carpeta, $2: patrón. Conserva los KEEP más recientes (el nombre lleva la fecha).
    find "$1" -maxdepth 1 -name "$2" | sort -r | tail -n +"$((KEEP + 1))" | while read -r old; do
        rm -f "$old"
        echo "Pruned $old"
    done
}
prune "$BACKUP_DIR/db" 'mlflow-*.dump'
for volume in $VOLUMES; do
    prune "$BACKUP_DIR/volumes" "$volume-*.tar.gz"
done

STEP=fin
echo "$STAMP" >"$BACKUP_DIR/last-success"
echo "Backup $STAMP OK in $BACKUP_DIR"
