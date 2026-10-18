# Entrenamiento continuo en Jenkins

Dos jobs. La lógica vive en los `.groovy` de esta carpeta: el job solo hace
`load` del archivo desde el despliegue, así que un cambio aquí rige en el
siguiente build sin tocar Jenkins.

| Job | Pipeline | Disparo |
|---|---|---|
| `house-price-release-sales` | `release-sales.groovy` | A mano: publica el mes siguiente de ventas (fuente simulada) |
| `house-price-ct` | `continuous-training.groovy` | Al terminar el anterior, y un cron diario |

`house-price-ct` reentrena solo si el dataset vivo no es con el que se entrenó
`@champion` (tag `dataset_sha256`). Si no hay datos nuevos pero la alerta
`DriftDeDatosDelModelo` está activa, termina **inestable**: sin precios reales
nuevos, reentrenar daría el mismo modelo. El drift se consulta a Prometheus en
cada build (no hay webhook de Alertmanager: Jenkins exige autenticación para
disparar builds y no hace falta exponer un token).

Con ventas nuevas, antes de reentrenar, el `@champion` tasa las casas de los
meses que no vio y se compara con el precio de venta
(`scripts/production_performance.py`): es su error real en producción. Queda
como run en MLflow (`stage = production_performance`), como tag
`real_error_AAAA-MM` de la versión y en `/metrics` de la API, con la alerta
`ErrorRealDelModeloAlto` si el champion que se sirve está degradado (MAPE por
encima del de su test + 15 % o sesgo de más de ±5 %, en `configs/model_config.yaml`).

Resultados:

- **Éxito, "vN en producción":** el challenger ganó, la API se reinició y sirve la versión nueva.
- **Inestable, "challenger perdió":** sigue el champion; la comparación queda en MLflow.
- **Inestable, "champion degradado, challenger perdió":** el modelo en producción ya
  falla con ventas reales y reentrenar no lo arregla; hay que revisarlo.
- **Fallido:** error del ciclo o puerta de calidad no superada.

## Avisos

`house-price-ct` manda un correo (`scripts/notify.py`) en cada resultado que
no sea "sin cambios": promoción, challenger perdido, champion degradado, drift
sin datos y fallos del ciclo. Lleva el estado del registro (champion,
challenger, último error real) y el enlace al build. Sale por el mailserver
propio con `no-reply-house-price@edqs.online` hacia `contacto@edqs.online`;
las variables `SMTP_*` y `NOTIFY_TO` están en el `.env` del despliegue. Si el
envío falla, el build lo anota en el log y sigue.

Las alertas de Prometheus del sistema (drift, error real, caída de la API)
llegan por correo desde Alertmanager (`~/infraestructura-core`, receptor
`house-price-correo`).

Probar la configuración SMTP:

```bash
docker compose run --rm --no-deps trainer python scripts/notify.py test
```

## Instalación

Jenkins (`~/jenkins`) necesita el despliegue montado en la misma ruta que en el
host, porque `docker compose` resuelve con ella los bind mounts y el `.env`:

```yaml
    volumes:
      - /home/ubuntu/house-price-predictor:/home/ubuntu/house-price-predictor
```

Los jobs se crean copiando las definiciones y reiniciando Jenkins:

```bash
for job in house-price-ct house-price-release-sales; do
  docker exec jenkins mkdir -p /var/jenkins_home/jobs/$job
  docker cp deployment/jenkins/$job.xml jenkins:/var/jenkins_home/jobs/$job/config.xml
done
cd ~/jenkins && docker compose up -d --force-recreate
```

El primer build de cada job registra sus propiedades (cron, parámetros); hasta
entonces `house-price-release-sales` no muestra el parámetro `INIT_UNTIL`.

## Arranque de la simulación

El dataset vivo (volumen `training-data`) empieza con las ventas hasta febrero de
2015, y el primer modelo se promueve sin comparar: el champion anterior entrenó
con el archivo completo, test incluido, así que la comparación no sería justa.

```bash
cd ~/house-price-predictor
docker compose run --rm trainer python scripts/release_sales.py --init 2015-02
docker compose run --rm -e PROMOTE=false trainer
docker compose run --rm trainer python scripts/promote_model.py --skip-comparison
docker compose restart house-price-api
```

Después, cada build de `house-price-release-sales` publica un mes (marzo, abril
y mayo de 2015) y `house-price-ct` decide.
