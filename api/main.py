from __future__ import annotations

from fastapi import FastAPI, HTTPException

from api.schemas import CustomerFeatures, PredictionResponse
from churn_prediction.predict import load_artifacts, predict_one

app = FastAPI(
    title="Customer Churn Prediction API",
    version="0.1.0",
    description="FastAPI service for predicting e-commerce customer churn risk.",
)

model = None
threshold = 0.5
metrics = {}


@app.on_event("startup")
def startup_event() -> None:
    global model, threshold, metrics
    try:
        model, threshold, metrics = load_artifacts()
    except FileNotFoundError:
        model = None


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": model is not None}


@app.get("/model-info")
def model_info() -> dict:
    return metrics or {"message": "No metrics artifact found. Train the model first."}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: CustomerFeatures) -> dict:
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train the model first.")
    return predict_one(payload.model_dump(), model=model, threshold=threshold)


@app.post("/predict_batch")
def predict_batch(payloads: list[CustomerFeatures]) -> list[dict]:
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train the model first.")
    return [predict_one(payload.model_dump(), model=model, threshold=threshold) for payload in payloads]
