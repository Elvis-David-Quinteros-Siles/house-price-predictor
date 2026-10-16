from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from inference import predict_price, batch_predict, model_info, metadata, InvalidInputError
import metrics
from schemas import HousePredictionRequest, PredictionResponse
from prometheus_fastapi_instrumentator import Instrumentator, metrics as http_metrics

# Initialize FastAPI app with metadata
app = FastAPI(
    title="House Price Prediction API",
    description=(
        "An API for predicting house prices based on various features. "
        "This application is part of the MLOps Bootcamp by School of Devops. "
        "Authored by Gourav Shah."
    ),
    version="1.0.0",
    contact={
        "name": "School of Devops",
        "url": "https://schoolofdevops.com",
        "email": "learn@schoolofdevops.com",
    },
    license_info={
        "name": "Apache 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
    },
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize and instrument prometheus metrics
# Buckets de latencia por endpoint afinados a una API que responde en decenas de
# ms (los de serie empiezan en 0,1 s y no permiten calcular un p95 útil).
instrumentator = Instrumentator().add(http_metrics.default(
    latency_lowr_buckets=(0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.25, 0.5, 1),
))
instrumentator.instrument(app).expose(app)
metrics.publish_model_metadata(metadata)

# Health check endpoint
@app.get("/health", response_model=dict)
async def health_check():
    return {"status": "healthy", "model_loaded": True}

# Prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
async def predict(request: HousePredictionRequest):
    try:
        response = predict_price(request)
    except InvalidInputError as e:
        metrics.record_rejection("predict", e.code)
        raise HTTPException(status_code=422, detail={"code": e.code, "message": str(e)})
    metrics.record("predict", [response.predicted_price])
    return response

# Batch prediction endpoint
@app.post("/batch-predict", response_model=list)
async def batch_predict_endpoint(requests: list[HousePredictionRequest]):
    try:
        prices = batch_predict(requests)
    except InvalidInputError as e:
        metrics.record_rejection("batch", e.code)
        raise HTTPException(status_code=422, detail={"code": e.code, "message": str(e)})
    metrics.record("batch", prices)
    return prices

# Served model metadata: metrics, interval and valid input values
@app.get("/model-info", response_model=dict)
async def get_model_info():
    return model_info()