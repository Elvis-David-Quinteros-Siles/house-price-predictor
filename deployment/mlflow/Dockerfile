# Servidor de tracking y registro de modelos de MLflow.
# Metadatos en Postgres; artefactos en el MinIO compartido (s3://house-price-mlflow).
# El servidor hace de proxy de artefactos (--serve-artifacts): el entrenamiento
# y la API solo necesitan la URL del servidor, nunca las credenciales de MinIO.
FROM python:3.11-slim

RUN pip install --no-cache-dir mlflow==3.16.1 psycopg2-binary==2.9.10 boto3==1.40.40

RUN useradd --create-home --uid 1000 mlflow \
 && mkdir /mlflow && chown mlflow /mlflow
# Datos locales (sqlite y artefactos) cuando no hay Postgres ni S3: docker-compose.yaml.
USER mlflow

EXPOSE 5000
