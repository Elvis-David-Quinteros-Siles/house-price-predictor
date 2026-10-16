# ── Etapa 1: pipeline de entrenamiento (datos → features → modelo)
# Los .pkl no se versionan: se generan aquí a partir de data/raw y de
# configs/model_config.yaml. MLflow registra en un file store local.
FROM python:3.11-slim AS train

WORKDIR /work

COPY src/api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt mlflow==2.3.1 setuptools==65.5.0

COPY configs/ configs/
COPY data/raw/ data/raw/
COPY src/ src/

RUN mkdir -p data/processed models/trained \
 && python src/data/run_processing.py \
 && python src/features/engineer.py \
      --input data/processed/cleaned_house_data.csv \
      --output data/processed/featured_house_data.csv \
      --preprocessor models/trained/preprocessor.pkl \
 && python src/models/train_model.py \
      --config configs/model_config.yaml \
      --data data/processed/featured_house_data.csv \
      --models-dir models \
      --mlflow-tracking-uri file:///work/mlruns

# ── Etapa 2: API de inferencia
FROM python:3.11-slim

WORKDIR /app
ENV PYTHONPATH=/app PYTHONUNBUFFERED=1

COPY src/api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --from=train /work/models/trained/house_price_model.pkl /work/models/trained/preprocessor.pkl /work/models/trained/model_metadata.json /app/models/trained/
COPY src/api/ /app/

RUN useradd --create-home --uid 1000 app
USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
