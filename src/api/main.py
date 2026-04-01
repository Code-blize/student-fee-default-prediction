from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from src.api.schemas import PredictionRequest, PredictionResponse
from src.features.build_features import build_feature_table
from src.models.predict import load_model, predict_single

ml_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = Path("models/best_model.pkl")
    ml_state["model"] = load_model(model_path)
    yield
    ml_state.clear()


app = FastAPI(
    title="Fee Defaulter Prediction API",
    version="1.0.0",
    description="API for predicting student fee-default risk.",
    lifespan=lifespan,
)


@app.get("/", response_class=HTMLResponse)
def root():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <title>Student Fee Defaulter Prediction API</title>
        <style>
            body {
                margin: 0;
                font-family: Arial, sans-serif;
                background: linear-gradient(135deg, #0f172a, #1e293b);
                color: white;
                display: flex;
                justify-content: center;
                align-items: center;
                min-height: 100vh;
            }
            .container {
                max-width: 850px;
                text-align: center;
                padding: 40px 24px;
            }
            h1 {
                font-size: 2.5rem;
                margin-bottom: 12px;
            }
            p {
                font-size: 1.1rem;
                line-height: 1.7;
                color: #cbd5e1;
                margin-bottom: 18px;
            }
            .badge {
                display: inline-block;
                padding: 8px 14px;
                border-radius: 999px;
                background: rgba(56, 189, 248, 0.15);
                color: #7dd3fc;
                font-size: 0.95rem;
                margin-bottom: 22px;
            }
            .buttons {
                margin-top: 28px;
            }
            a {
                text-decoration: none;
                display: inline-block;
                margin: 10px;
                padding: 12px 22px;
                border-radius: 10px;
                font-weight: bold;
                transition: 0.3s ease;
            }
            .btn-primary {
                background: #38bdf8;
                color: #0f172a;
            }
            .btn-primary:hover {
                background: #0ea5e9;
            }
            .btn-secondary {
                background: transparent;
                border: 1px solid #94a3b8;
                color: white;
            }
            .btn-secondary:hover {
                background: rgba(255, 255, 255, 0.08);
            }
            .info-box {
                margin-top: 28px;
                padding: 18px;
                border-radius: 14px;
                background: rgba(255, 255, 255, 0.06);
                color: #e2e8f0;
                text-align: left;
            }
            .info-box ul {
                margin: 10px 0 0 18px;
                padding: 0;
            }
            .footer {
                margin-top: 28px;
                font-size: 0.95rem;
                color: #94a3b8;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="badge">FastAPI • Machine Learning Inference • Render Deployment</div>
            <h1>Student Fee Defaulter Prediction API</h1>
            <p>
                An end-to-end machine learning API for predicting student fee-default risk
                using academic, family, and support-related indicators.
            </p>

            <div class="buttons">
                <a href="/docs" class="btn-primary">Open API Docs</a>
                <a href="/health" class="btn-secondary">Health Check</a>
            </div>

            <div class="info-box">
                <strong>Available endpoints:</strong>
                <ul>
                    <li><code>/</code> — landing page</li>
                    <li><code>/health</code> — service health check</li>
                    <li><code>/docs</code> — interactive API documentation</li>
                    <li><code>/predict</code> — prediction endpoint</li>
                </ul>
            </div>

            <div class="footer">
                Built by Blessing Obasi-Uzoma
            </div>
        </div>
    </body>
    </html>
    """


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    raw_df = pd.DataFrame([request.model_dump()])
    feature_df = build_feature_table(raw_df)

    model = ml_state["model"]
    result = predict_single(feature_df, model)

    probability = result["probability"]

    if probability >= 0.30:
        risk_label = "High Risk"
    elif probability >= 0.20:
        risk_label = "Moderate Risk"
    else:
        risk_label = "Low Risk"

    return PredictionResponse(
        prediction=result["prediction"],
        probability=probability,
        risk_label=risk_label,
    )
