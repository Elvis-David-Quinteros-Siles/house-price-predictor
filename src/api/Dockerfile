# Dos imágenes desde el mismo Dockerfile, con las mismas dependencias:
#   --target train  entrenador: datos -> modelo en el registro de MLflow
#                   (scripts/train.sh). El modelo ya no se entrena en el build.
#   (por defecto)   API de inferencia: carga MODEL_URI del registro al arrancar.

# ── Base: dependencias comunes
FROM python:3.11-slim AS base

# LightGBM necesita la librería de OpenMP.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# Sin git en la imagen: el commit llega por GIT_SHA y MLflow no debe buscarlo.
# Artefactos siempre a través del servidor MLflow: con S3 detrás, el cliente
# pediría URLs prefirmadas de MinIO, al que la API y el entrenador no llegan.
ENV PYTHONUNBUFFERED=1 \
    GIT_PYTHON_REFRESH=quiet \
    MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD=false \
    MLFLOW_ENABLE_PROXY_MULTIPART_UPLOAD=false

COPY src/api/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

RUN useradd --create-home --uid 1000 app

# ── Entrenador
FROM base AS train

WORKDIR /work
COPY configs/ configs/
COPY data/raw/ data/raw/
COPY src/ src/
COPY scripts/ scripts/
RUN mkdir -p data/processed models/trained && chown -R app data/processed models

# Commit del código, que queda como tag de cada versión del modelo.
ARG GIT_SHA=unknown
ENV GIT_SHA=${GIT_SHA}

USER app
CMD ["sh", "scripts/train.sh"]

# ── API de inferencia
FROM base

WORKDIR /app
ENV PYTHONPATH=/app \
    MODEL_URI=models:/house_price_model@champion

COPY src/api/ /app/
# El pipeline del modelo referencia las clases de engineer.py.
COPY src/features/engineer.py /app/
# Caché de la versión descargada (un volumen en el despliegue).
RUN mkdir -p /app/models/cache && chown app /app/models/cache

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
