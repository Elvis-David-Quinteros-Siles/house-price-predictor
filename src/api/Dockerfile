# ── Etapa 1: pipeline de entrenamiento (datos → pipeline de variables + modelo)
# Los .pkl no se versionan: se generan aquí a partir de data/raw y de
# configs/model_config.yaml. MLflow registra en un file store local.
FROM python:3.11-slim AS train

# LightGBM necesita la librería de OpenMP.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /work

COPY src/api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt mlflow==2.3.1 setuptools==65.5.0

COPY configs/ configs/
COPY data/raw/ data/raw/
COPY src/ src/

RUN mkdir -p data/processed models/trained \
 && python src/data/run_processing.py \
 && python src/models/train_model.py \
      --config configs/model_config.yaml \
      --data data/processed/cleaned_house_data.csv \
      --models-dir models \
      --mlflow-tracking-uri file:///work/mlruns

# ── Etapa 2: API de inferencia
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV PYTHONPATH=/app PYTHONUNBUFFERED=1

COPY src/api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --from=train /work/models/trained/house_price_model.pkl /work/models/trained/model_metadata.json /work/models/trained/sold_locations.npy /app/models/trained/
COPY src/api/ /app/
# El .pkl del pipeline referencia las clases de engineer.py.
COPY src/features/engineer.py /app/

RUN useradd --create-home --uid 1000 app
USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
