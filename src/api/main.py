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


from fastapi.responses import HTMLResponse

@app.get("/", response_class=HTMLResponse)
def root():
    return """
    <html>
      <body>
        <h1>Student Fee Defaulter Prediction API</h1>
      </body>
    </html>
    """


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def root():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Student Fee Defaulter Prediction API</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                background: linear-gradient(135deg, #0f172a, #1e293b);
                color: white;
                margin: 0;
                padding: 0;
                display: flex;
                justify-content: center;
                align-items: center;
                min-height: 100vh;
            }
            .container {
                max-width: 800px;
                text-align: center;
                padding: 40px;
            }
            h1 {
                font-size: 2.5rem;
                margin-bottom: 10px;
            }
            p {
                font-size: 1.1rem;
                line-height: 1.6;
                color: #cbd5e1;
            }
            .buttons {
                margin-top: 30px;
            }
            a {
                text-decoration: none;
                display: inline-block;
                margin: 10px;
                padding: 12px 24px;
                border-radius: 10px;
                font-weight: bold;
                transition: 0.3s;
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
                background: rgba(255,255,255,0.08);
            }
            .footer {
                margin-top: 30px;
                font-size: 0.95rem;
                color: #94a3b8;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Student Fee Defaulter Prediction API</h1>
            <p>
                An end-to-end machine learning API for predicting student fee-default risk
                using academic, family, and support-related indicators.
            </p>
            <div class="buttons">
                <a href="/docs" class="btn-primary">Open API Docs</a>
                <a href="/health" class="btn-secondary">Health Check</a>
            </div>
            <div class="footer">
                Built with FastAPI • Model Inference Endpoint • Render Deployment
            </div>
        </div>
    </body>
    </html>
    """
