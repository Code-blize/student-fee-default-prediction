from __future__ import annotations

from pathlib import Path
import joblib
import pandas as pd


def load_model(model_path: str | Path):
    """Load a trained model artifact."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    return joblib.load(model_path)


def align_features(df: pd.DataFrame, model) -> pd.DataFrame:
    """Align input features to the order expected by the model."""
    if not hasattr(model, "feature_names_in_"):
        return df

    required_features = list(model.feature_names_in_)

    missing = [col for col in required_features if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required prediction features: {missing}")

    return df[required_features].copy()


def predict_batch(df: pd.DataFrame, model) -> pd.DataFrame:
    """Generate batch predictions and probabilities."""
    X = align_features(df, model)

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    result = X.copy()
    result["prediction"] = predictions
    result["probability"] = probabilities

    return result


def predict_single(record: dict, model) -> dict:
    """Generate prediction for a single record."""
    df = pd.DataFrame([record])
    result = predict_batch(df, model)
    row = result.iloc[0]

    return {
        "prediction": int(row["prediction"]),
        "probability": float(row["probability"]),
    }
