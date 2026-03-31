from __future__ import annotations

from pathlib import Path
import pandas as pd
from fastapi import FastAPI

from src.api.schemas import PredictionRequest, PredictionResponse
from src.features.build_features import build_feature_table
from src.models.predict import load_model, predict_single


app = FastAPI(
    title="Fee Defaulter Prediction API",
    version="1.0.0",
    description="API for predicting student fee-default risk."
)

MODEL_PATH = Path("models/best_model.pkl")
model = load_model(MODEL_PATH)


@app.get("/")
def root():
    return {
        "message": "Fee Defaulter Prediction API is running.",
        "docs": "/docs"
    }


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    raw_df = pd.DataFrame([request.model_dump()])

    feature_df = build_feature_table(raw_df)

    result = predict_single(feature_df, model)

    probability = result["probability"]
    prediction = result["prediction"]

    if probability >= 0.30:
        risk_label = "High Risk"
    elif probability >= 0.20:
        risk_label = "Moderate Risk"
    else:
        risk_label = "Low Risk"

    return PredictionResponse(
        prediction=prediction,
        probability=probability,
        risk_label=risk_label
    )
