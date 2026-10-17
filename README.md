# 🏠 House Price Predictor – An MLOps Learning Project

Welcome to the **House Price Predictor** project! This is a real-world, end-to-end MLOps use case designed to help you master the art of building and operationalizing machine learning pipelines.

You'll start from raw data and move through data preprocessing, feature engineering, experimentation, model tracking with MLflow, and optionally using Jupyter for exploration – all while applying industry-grade tooling.


## 📦 Project Structure

```
house-price-predictor/
├── configs/                # YAML-based configuration for models
├── data/                   # Raw and processed datasets
├── deployment/
│   └── mlflow/             # Docker Compose setup for MLflow
├── models/                 # Trained models and preprocessors
├── notebooks/              # Optional Jupyter notebooks for experimentation
├── src/
│   ├── data/               # Data cleaning and preprocessing scripts
│   ├── features/           # Feature engineering pipeline
│   ├── models/             # Model training and evaluation
├── requirements.txt        # Python dependencies
└── README.md               # You’re here!
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

## 📊 Setup MLflow for Experiment Tracking

To track experiments and model runs:

```bash
cd deployment/mlflow
docker compose -f mlflow-docker-compose.yml up -d
docker compose ps
```

> 🐧 **Using Podman?** Use this instead:

```bash
podman compose -f mlflow-docker-compose.yml up -d
podman compose ps
```

Access the MLflow UI at [http://localhost:5555](http://localhost:5555)

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
- **Served model:** trained with a fixed seed (`random_state: 42`), so every build produces the same model. On the 20% hold-out test set it scores R² 0.91, MAE $61.5k and MAPE 11.1% (5-fold CV on the training split: R² 0.90, MAE $60.3k); these numbers are written to `models/trained/model_metadata.json` and exposed by `GET /model-info`.
- **Price range:** an empirical 90% interval taken from the cross-validation (out-of-fold) predictions: 90% of those houses sold between 0.78× and 1.26× the predicted price, and the API applies those ratios to each prediction. On the untouched test set the interval holds 90.5% of the houses.
- **Location check:** the API rejects points more than 0.5 km from any sale in the dataset (lakes, Puget Sound, outside the county); 99% of the sales have another one within 0.55 km.
- **Interface:** Streamlit in Spanish and English (selector at the top, or `?lang=en` in the URL), with an OpenStreetMap map to place the house.

## 🔁 Model Workflow

### 🧹 Step 1: Data Processing

Clean and preprocess the raw housing dataset:

```bash
python src/data/run_processing.py   --input data/raw/kc_house_data.csv   --output data/processed/cleaned_house_data.csv
```

---

### 🧠 Step 2: Train the model

The features, the preprocessing and the model form a single scikit-learn
pipeline (`src/features/engineer.py`), fitted on the training split only. The
script cross-validates it, evaluates it on the test split and logs everything to
MLflow:

```bash
python src/models/train_model.py   --config configs/model_config.yaml   --data data/processed/cleaned_house_data.csv   --models-dir models   --mlflow-tracking-uri http://localhost:5555
```

---


## Building FastAPI and Streamlit 

The code for both the apps are available in `src/api` and `streamlit_app` already. To build and launch these apps 

  * Add a  `Dockerfile` in the root of the source code for building FastAPI  
  * Add `streamlit_app/Dockerfile` to package and build the Streamlit app  
  * Add `docker-compose.yaml` in the root path to launch both these apps. be sure to provide `API_URL=http://fastapi:8000` in the streamlit app's environment. 


Once you have launched both the apps, you should be able to access streamlit web ui and make predictions. 

You could also test predictions with FastAPI directly using 

```
curl -X POST "http://localhost:8000/predict" \
-H "Content-Type: application/json" \
-d '{
  "sqft": 1500,
  "bedrooms": 3,
  "bathrooms": 2,
  "location": "suburban",
  "year_built": 2000,
  "condition": fair
}'

```

Be sure to replace `http://localhost:8000/predict` with actual endpoint based on where its running. 


## ✅ Tests and CI

```bash
# Unit tests of the data and feature pipeline
pip install -r src/api/requirements.txt -r tests/requirements.txt
python -m pytest tests/unit

# API tests against a running API (default http://localhost:8000)
API_URL=http://localhost:8000 python -m pytest tests/api
```

`.github/workflows/ci.yml` runs on every push and pull request to `main`:

1. Unit tests.
2. Builds the API image, which trains the model.
3. **Quality gate:** `scripts/check_quality_gate.py` fails the build if the model's test metrics are worse than the minimums in `configs/model_config.yaml` (`quality_gate`).
4. Starts the API and runs the API tests against it: validation, location checks, prices moving in the expected direction.
5. Builds the Streamlit image and, on `main`, publishes both images to GitHub Container Registry (`ghcr.io/<owner>/house-price-api` and `house-price-predictor`).

## 🧠 Learn More About MLOps

- Build and track ML pipelines
- Containerize and deploy models
- Automate training workflows using GitHub Actions or Argo Workflows
- Apply DevOps principles to Machine Learning systems


---

