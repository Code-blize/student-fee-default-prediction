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
    """
    Align input features to the exact columns expected by the trained model.
    Missing columns are added with 0, extra columns are removed.
    """
    if not hasattr(model, "feature_names_in_"):
        return df.copy()

    required_features = list(model.feature_names_in_)

    aligned_df = df.copy()

    for col in required_features:
        if col not in aligned_df.columns:
            aligned_df[col] = 0

    aligned_df = aligned_df[required_features]

    return aligned_df


def predict_batch(df: pd.DataFrame, model) -> pd.DataFrame:
    """Generate batch predictions and probabilities."""
    X = align_features(df, model)

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    result = X.copy()
    result["prediction"] = predictions
    result["probability"] = probabilities

    return result


def predict_single(df: pd.DataFrame, model) -> dict:
    """Generate prediction for a single prepared dataframe row."""
    result = predict_batch(df, model)
    row = result.iloc[0]

    return {
        "prediction": int(row["prediction"]),
        "probability": float(row["probability"]),
    }
