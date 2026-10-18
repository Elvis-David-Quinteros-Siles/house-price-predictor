# Backups del registro de modelos

Sin estos backups, perder el volumen de Postgres o el bucket de MinIO es
perder todas las versiones del modelo, los alias `@champion`/`@challenger` y
el historial de entrenamientos, comparaciones y error real.

| Qué | Cómo | Dónde (`~/backups/house-price/`) | Retención |
|---|---|---|---|
| Base de MLflow (`house-price-mlflow-db`) | `pg_dump --format=custom`, verificado con `pg_restore --list` | `db/mlflow-AAAAMMDD-HHMMSS.dump` | 14 |
| Artefactos (bucket `house-price-mlflow`) | `mc mirror`, acumulativo: nunca borra | `artifacts/` | todo |
| Volúmenes `prediction-log` y `training-data` | `tar czf` | `volumes/<volumen>-AAAAMMDD-HHMMSS.tar.gz` | 14 |

`model-cache` y `model-performance` no se copian: se regeneran desde el
registro y con el siguiente ciclo de entrenamiento continuo.

## Programación (cron del host, usuario ubuntu)

```cron
30 3 * * * /home/ubuntu/house-price-predictor/app/deployment/backup/backup-registry.sh >> /home/ubuntu/backups/house-price/backup.log 2>&1
15 4 1 * * /home/ubuntu/house-price-predictor/app/deployment/backup/restore-registry.sh --check >> /home/ubuntu/backups/house-price/backup.log 2>&1
```

Cron y no Jenkins: el backup no depende de que Jenkins esté vivo. Si el backup
o la prueba de restauración fallan, llega un correo (`scripts/notify.py
backup_failed`). El último backup bueno queda en `last-success`.

## Restaurar

```bash
cd ~/house-price-predictor/app/deployment/backup
./restore-registry.sh --check            # prueba con el último dump, sin tocar producción
./restore-registry.sh --yes              # restauración real con el último dump
./restore-registry.sh --yes ~/backups/house-price/db/mlflow-20261017-220400.dump
```

La restauración real guarda primero la base actual en
`db/pre-restore-*.dump`, para deshacerla con el mismo script; para MLflow,
reemplaza la base, repone en el bucket los artefactos que falten, comprueba
que cada versión tenga su modelo y reinicia la API para que cargue el
`@champion` restaurado.

Los volúmenes se restauran a mano, con el servicio que los usa parado:

```bash
docker run --rm -v house-price-predictor_training-data:/data \
  -v ~/backups/house-price/volumes:/backup:ro postgres:16-alpine \
  sh -c 'rm -rf /data/* && tar xzf /backup/training-data-AAAAMMDD-HHMMSS.tar.gz -C /data'
```

## Límite

Los backups están en el mismo disco que los datos: protegen de errores
(un volumen borrado, una migración que sale mal, un objeto eliminado), no de
perder el servidor. Para eso hace falta una copia fuera (otro servidor,
almacenamiento externo) de `~/backups/house-price`.
