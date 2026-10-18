# 🏠 House Price Predictor – An MLOps Learning Project

Welcome to the **House Price Predictor** project! This is a real-world, end-to-end MLOps use case designed to help you master the art of building and operationalizing machine learning pipelines.

You'll start from raw data and move through data preprocessing, feature engineering, experimentation, model tracking with MLflow, and optionally using Jupyter for exploration – all while applying industry-grade tooling.


The model runs in production with the full MLOps loop: a model registry, a quality gate, champion vs challenger, data drift and real-error monitoring, continuous training in Jenkins, email notifications and backups.

> 📘 **How it all works in production (in Spanish):** [`docs/MLOPS.md`](docs/MLOPS.md), covering the architecture, every stage of the lifecycle, metrics and alerts, the runbook, thresholds and design decisions.

## 📦 Project Structure

```
house-price-predictor/
├── configs/model_config.yaml   # Model, quality gate, champion vs challenger and real-error thresholds
├── data/raw/                   # King County dataset
├── deployment/
│   ├── backup/                 # Registry backup and restore scripts (cron)
│   ├── jenkins/                # Continuous training pipelines
│   ├── kubernetes/             # Kubernetes manifests
│   └── mlflow/                 # MLflow tracking server image
├── docs/MLOPS.md               # Production MLOps, end to end
├── notebooks/                  # Optional Jupyter notebooks for experimentation
├── scripts/
│   ├── train.sh                # Process -> train -> promote (trainer image command)
│   ├── promote_model.py        # Quality gate + champion vs challenger -> @champion
│   ├── compare_models.py       # Both versions on the same test set
│   ├── check_quality_gate.py   # Absolute minimums on the test metrics
│   ├── release_sales.py        # Simulated data source: one month of sales at a time
│   ├── continuous_training.py  # Checks for the CT job: new data? drift? champion?
│   ├── production_performance.py  # Real error of @champion on newly released sales
│   └── notify.py               # Email notifications
├── src/
│   ├── api/                    # FastAPI service (+ Dockerfile for the API and trainer images)
│   ├── data/                   # Data cleaning
│   ├── features/               # Feature engineering pipeline
│   ├── models/                 # Training and registration
│   └── monitoring/             # Data drift (PSI) and real error
├── streamlit_app/              # Web interface (Spanish/English, map)
├── tests/                      # Unit and API tests
└── README.md                   # You’re here!
```

---

## 🛠️ Setting up Learning/Development Environment

To begin, ensure the following tools are installed on your system:

- [Python 3.11](https://www.python.org/downloads/)
- [Git](https://git-scm.com/)
- [Visual Studio Code](https://code.visualstudio.com/) or your preferred editor
- [UV – Python package and environment manager](https://github.com/astral-sh/uv)
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) **or** [Podman Desktop](https://podman-desktop.io/)

---

## 🚀 Preparing Your Environment

1. **Fork this repo** on GitHub.

2. **Clone your forked copy:**

   ```bash
   # Replace xxxxxx with your GitHub username or org
   git clone https://github.com/xxxxxx/house-price-predictor.git
   cd house-price-predictor
   ```

3. **Setup Python Virtual Environment using UV:**

   ```bash
   uv venv --python python3.11
   source .venv/bin/activate
   ```

4. **Install dependencies:**

   ```bash
   uv pip install -r requirements.txt
   ```

---

## 📊 MLflow: tracking and model registry

Every trained model goes to the MLflow model registry, and the API serves the
version with the alias **`@champion`**. Locally, `docker-compose.yaml` runs an
MLflow server with sqlite and the artifacts in a Docker volume:

```bash
docker compose up -d mlflow          # UI at http://localhost:5555
docker compose run --rm trainer      # train, register and promote to @champion
docker compose up -d                 # API (loads @champion) and Streamlit
```

In production (`deployment/mlflow/Dockerfile`), metadata live in Postgres and
artifacts in S3 (MinIO). The server proxies the artifacts
(`--artifacts-destination`), so the trainer and the API only need
`MLFLOW_TRACKING_URI`, never storage credentials.

---

## 📒 Using JupyterLab (Optional)

If you prefer an interactive experience, launch JupyterLab with:

```bash
uv python -m jupyterlab
# or
python -m jupyterlab
```

---

## 🏠 Dataset and model

- **Data:** [House Sales in King County, USA](https://www.kaggle.com/datasets/harlfoxem/housesalesprediction) (CC0): 21,613 real sales in the Seattle area between May 2014 and May 2015, stored in `data/raw/kc_house_data.csv`. Predicted prices are therefore in 2014–2015 dollars.
- **Cleaning:** repeated sales of the same house keep only the latest one (so a house cannot be in train and test at once) and 18 rows with impossible bedroom/bathroom counts are dropped, leaving 21,418 sales.
- **Inputs:** location (latitude/longitude, picked on a map), living area, lot size, bedrooms, bathrooms, year built, condition (King County scale: Poor → Very Good), construction grade (1–13, 7 = average), view (0–4) and waterfront.
- **Model selection** (5-fold cross-validation). First the algorithm, with the original six inputs:

| Model | R² | MAE | MAPE |
|---|---|---|---|
| Baseline (median) | -0.06 | $221.7k | 42.1% |
| Linear regression | 0.75 | $108.7k | 21.3% |
| Gradient Boosting, 100 trees, depth 3 | 0.74 | $115.9k | 23.9% |
| Random Forest, 300 trees | 0.74 | $98.3k | 18.1% |
| Gradient Boosting, 600 trees, depth 5 | 0.78 | $90.3k | 16.6% |
| **Gradient Boosting, 600 trees, depth 5, log(price)** | **0.80** | **$87.0k** | **15.4%** |

  Then the features, with that algorithm:

| Features | R² | MAE | MAPE |
|---|---|---|---|
| Original six (zipcode as location) | 0.80 | $87.1k | 15.4% |
| + grade, view, waterfront, lot size (zipcode) | 0.87 | $72.7k | 13.4% |
| + grade, view, waterfront, lot size (latitude/longitude) | 0.90 | $64.9k | 11.9% |
| + the same, latitude/longitude and zipcode | 0.90 | $63.5k | 11.7% |

  Coordinates beat the zipcode, and adding the zipcode on top is within the noise (±0.008 R²), so the model uses coordinates only.

  Finally, with the API inputs, other boosting libraries and two extra features:

| Model and features | R² | MAE | MAPE |
|---|---|---|---|
| Gradient Boosting, 600 trees, depth 5 | 0.89 | $65.6k | 12.1% |
| XGBoost | 0.90 | $63.7k | 11.8% |
| LightGBM | 0.90 | $62.6k | 11.6% |
| **LightGBM + sale month + prices of the 10 nearest sales** | **0.91** | **$59.9k** | **11.2%** |

  The neighbour features are the mean log price, log price per sq ft, living area and distance of the 10 nearest sales in the training set (each house is left out of its own neighbours). The sale month captures the 2014–2015 price rise; predictions use the latest month. Hyperparameters were tuned with Optuna (44 trials), which only gained $0.1k over hand-set values. Asking for floors, basement and renovation year would bring the MAE to $58.6k, but the API does not take those inputs yet.
- **One pipeline:** features, preprocessing and model are a single scikit-learn pipeline (`src/features/engineer.py`) fitted on the training split only and saved as one `.pkl`, so training and the API compute features with the same code.
- **Served model:** trained with a fixed seed (`random_state: 42`), so every training run on the same data produces the same model. Trained on the whole archive it scores R² 0.91, MAE $61.5k and MAPE 11.1% on the 20% hold-out test set. In production the model is retrained as simulated sales arrive (see [Continuous training](#-continuous-training)). The current `@champion` is v5, trained on sales up to March 2015: R² 0.917, MAE $56.9k, MAPE 10.9%. The metrics of the served version are in `serving/model_metadata.json` and exposed by `GET /model-info`.
- **Price range:** an empirical 90% interval taken from the cross-validation (out-of-fold) predictions. For v5, 90% of those houses sold between 0.77× and 1.27× the predicted price, and the API applies those ratios to each prediction. On the untouched test set the interval holds 91.8% of the houses.
- **Location check:** the API rejects points more than 0.5 km from any sale in the dataset (lakes, Puget Sound, outside the county); 99% of the sales have another one within 0.55 km.
- **Interface:** Streamlit in Spanish and English (selector at the top, or `?lang=en` in the URL), with an OpenStreetMap map to place the house.

## 🔁 Model Workflow

### 🧹 Step 1: Data Processing

Clean and preprocess the raw housing dataset:

```bash
python src/data/run_processing.py   --input data/raw/kc_house_data.csv   --output data/processed/cleaned_house_data.csv
```

---

### 🧠 Step 2: Train and register the model

The features, the preprocessing and the model form a single scikit-learn
pipeline (`src/features/engineer.py`), fitted on the training split only. The
script cross-validates it, evaluates it on the test split and registers it in
MLflow as a new version of `house_price_model` with the alias `@challenger`:

```bash
export MLFLOW_TRACKING_URI=http://localhost:5555
python src/models/train_model.py   --config configs/model_config.yaml   --data data/processed/cleaned_house_data.csv   --models-dir models
```

Each version carries its lineage as tags: commit (`git_sha`), hash of the raw
dataset (`dataset_sha256`) and library versions. Next to the model, the run
stores what the API needs to serve it (`serving/model_metadata.json`,
`serving/sold_locations.npy` and `serving/reference_profile.json`, the
training-data profile used to measure drift) and its test split
(`evaluation/holdout.csv`).

### 🚦 Step 3: Promote to production

```bash
python scripts/promote_model.py                          # @challenger -> @champion if it passes both checks
python scripts/promote_model.py --version 3              # roll back to a previous version (no comparison)
```

The challenger has to pass two checks:

1. **Quality gate:** absolute minimums on its own test metrics (`quality_gate` in `configs/model_config.yaml`).
2. **Champion vs challenger:** both versions predict the challenger's test split (`evaluation/holdout.csv`), so they are scored on the same houses. The challenger wins if its MAE is no more than `max_mae_increase` (1 %) above the champion's (`champion_challenger`). Otherwise it stays `@challenger` and the script exits with an error. `--skip-comparison` overrides the comparison.

Each comparison is logged as a run `vN vs vM` in the model's experiment, and the challenger version gets a `vs_champion` tag with the result.
`scripts/train.sh` chains the three steps; it is the trainer image's command
(`docker build --target train -f src/api/Dockerfile .`).

### 🔁 Continuous training

The King County dataset is fixed (May 2014 - May 2015), so new sales are
simulated: `scripts/release_sales.py` publishes the archive month by month into
a live dataset, which the deployed trainer uses (`DATASET`). The train/test
split is stable per house (hash of its `id`): as the dataset grows, test houses
stay in test and no version ever trains on another version's test set, so
champion vs challenger stays a clean comparison.

Two Jenkins jobs run the loop (`deployment/jenkins/`):

- `house-price-release-sales` publishes the next month and triggers `house-price-ct`.
- `house-price-ct` (also runs daily) retrains only if the live dataset differs from the one `@champion` was trained on (`dataset_sha256` tag). Before retraining it measures the champion's [real error](#-real-error-against-sale-prices) on the new sales. Then it runs the quality gate and champion vs challenger, restarts the API and checks that it serves the new `@champion`. If the drift alert is firing but there are no new labelled sales, it ends *unstable*: retraining on the same data would give the same model.

Every outcome except "no changes" sends an [email](#-notifications).

### 🛰️ Serving

The API downloads `MODEL_URI` (default `models:/house_price_model@champion`)
from `MLFLOW_TRACKING_URI` at startup and keeps a copy in `models/cache`: if the
registry is down, it serves the last version it downloaded. To serve a newly
promoted version, restart the API. `GET /health`, `GET /model-info` and the
`house_price_model_info` metric report the version being served. Without
`MODEL_URI`, the API reads the files `train_model.py` leaves in `models/trained`.

### 📉 Drift monitoring

Every priced house (inputs, predicted price, model version, time) is appended
to a monthly JSONL log in `PREDICTION_LOG_DIR` (`data/predictions`, a volume in
production). Over the last `DRIFT_WINDOW` houses (default 500) the API computes
the PSI (population stability index) of each input and of the predicted price
against the training data of the served model (`src/monitoring/drift.py`, on
training deciles) and exposes it in `/metrics` as
`house_price_feature_drift_psi{feature}`, with the window size in
`house_price_drift_window_samples`. The window is reloaded from the log at
startup.

PSI below 0.1 is stable, 0.1–0.25 a moderate shift and above 0.25 a large one.
With few houses the PSI stays above zero even without drift (about
(bins − 1) / n: 0.09 with 100 houses), so the `DriftDeDatosDelModelo` alert
(PSI > 0.25 for 1 h) needs at least 100 houses in the window.

Running `tests/api` against the production API writes its test houses to the log.

### 🎯 Real error against sale prices

Drift only says that the inputs changed. To know whether the model is
*right*, it needs the real price, which arrives later. When a month of sales
is released, and before retraining, `scripts/production_performance.py`
makes `@champion` price those houses. All of them are after its
`data_until` tag, so the model never saw them. It then compares the
predictions with the sale prices (`src/monitoring/performance.py`): MAE,
MAPE, **bias** (median of (predicted − sold) / sold, negative when the model
prices below the market) and how many sales fall inside the 90% interval.

The model counts as **degraded** if, over at least 200 sales, its real MAPE
is more than 15% above its test MAPE or its bias is beyond ±5%
(`production_performance` in `configs/model_config.yaml`). For calibration,
v4 (sales up to February 2015, test MAPE 10.8%) was scored on the following
months without being retrained:

| Month | MAPE | Bias | In the 90% interval | Degraded |
|---|---|---|---|---|
| 2015-03 | 11.9% | −3.5% | 88.4% | no |
| 2015-04 | 12.3% | −5.8% | 87.9% | yes (bias) |
| 2015-05 | 13.3% | −6.6% | 83.0% | yes (MAPE and bias) |

The growing negative bias is concept drift: prices keep rising and a frozen
model falls behind. Monthly retraining corrects it.

Each evaluation is logged to MLflow, both as a `production_performance` run
and as a `real_error_YYYY-MM` tag on the version. The latest one also goes
into `/metrics` as `house_price_real_*`: the API re-reads it on every scrape
from a volume the trainer writes. The `ErrorRealDelModeloAlto` alert fires
only while the degraded version is still the one being served.

```bash
python scripts/production_performance.py                     # live dataset ($DATASET)
python scripts/production_performance.py --dataset data/raw/kc_house_data.csv --dry-run
```

### 📬 Notifications

`scripts/notify.py` emails the outcome of the continuous training job, with
the state of the registry (champion, challenger, comparison, last real error)
and a link to the build. The events are:

- a new model in production
- the challenger lost
- the champion is degraded and the challenger does not beat it
- drift without new data
- a failed run
- a failed backup

SMTP settings come from the environment (`SMTP_HOST`, `SMTP_PORT`,
`SMTP_USER`, `SMTP_PASSWORD`, `NOTIFY_TO`). Without `SMTP_HOST` nothing is
sent. A failed send never fails the job. In production, Alertmanager also
emails the Prometheus alerts of the system (drift, real error, API down).

```bash
python scripts/notify.py test     # check the SMTP settings
```

### 💾 Backups

`deployment/backup/backup-registry.sh` runs daily from the host's cron. It
takes three copies:

- a verified `pg_dump` of the MLflow database, keeping the last 14
- a cumulative mirror of the artifacts bucket, which never deletes
- tarballs of the prediction log and the live dataset

`restore-registry.sh --check` restores the latest dump into a throwaway
Postgres and checks that every model version has its files in the mirror. It
runs monthly and does not touch production. `restore-registry.sh --yes`
performs the real restore, saving the current database first. Failures send
an email. See [`deployment/backup/README.md`](deployment/backup/README.md).

---


## 🖥️ Running the API and the interface

`docker-compose.yaml` runs the whole stack locally: MLflow, the trainer, the
FastAPI service (`src/api`, port 8000) and the Streamlit interface
(`streamlit_app`, port 8501). The API loads `@champion` from MLflow at
startup, so train and promote a model first:

```bash
docker compose up -d mlflow
docker compose run --rm trainer      # train, register and promote to @champion
docker compose up -d fastapi streamlit
```

Open http://localhost:8501, or call the API directly:

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "sqft": 1800, "sqft_lot": 5000, "bedrooms": 3, "bathrooms": 2,
    "year_built": 1975, "condition": "Average", "grade": 7,
    "view": 0, "waterfront": false, "lat": 47.66, "long": -122.33
  }'
```

`GET /model-info` lists the served version, its metrics and the valid
values of each input. `GET /metrics` exposes the Prometheus metrics. In
production the API is only reachable by Streamlit and Prometheus, and the
interface is published at https://house.edqs.online.

## ✅ Tests and CI

```bash
# Unit tests: data, features, drift, real error, model comparison, continuous training, notifications
pip install -r src/api/requirements.txt -r tests/requirements.txt
python -m pytest tests/unit

# API tests against a running API (default http://localhost:8000)
API_URL=http://localhost:8000 python -m pytest tests/api
```

`.github/workflows/ci.yml` runs on every push and pull request to `main`:

1. Unit tests.
2. Builds the MLflow server, trainer and API images and starts a throwaway MLflow server.
3. Trains the model and registers it; **quality gate:** promoting it to `@champion` fails the build if its test metrics are worse than the minimums in `configs/model_config.yaml` (`quality_gate`). CI starts with an empty registry, so there is no champion to compare against.
4. Starts the API, which loads `@champion` from the registry as in production, and runs the API tests against it: validation, location checks, prices moving in the expected direction.
5. Builds the Streamlit image and, on `main`, publishes the images to GitHub Container Registry (`ghcr.io/<owner>/house-price-api`, `house-price-trainer`, `house-price-mlflow` and `house-price-predictor`).

## 🧠 Learn More About MLOps

- Build and track ML pipelines
- Containerize and deploy models
- Automate training workflows using GitHub Actions or Argo Workflows
- Apply DevOps principles to Machine Learning systems


---

