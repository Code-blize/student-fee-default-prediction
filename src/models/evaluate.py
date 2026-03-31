from __future__ import annotations

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


def classification_metrics(y_true, y_pred, y_prob) -> dict:
    """Compute core binary classification metrics."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
    }


def evaluate_thresholds(y_true, y_prob, thresholds: np.ndarray | None = None) -> pd.DataFrame:
    """Evaluate metrics across many probability thresholds."""
    if thresholds is None:
        thresholds = np.arange(0.1, 0.91, 0.05)

    rows = []
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)

        rows.append({
            "threshold": round(float(threshold), 2),
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "predicted_positive_count": int(y_pred.sum()),
        })

    return pd.DataFrame(rows)


def pick_best_f1_threshold(threshold_results: pd.DataFrame) -> pd.Series:
    """Return the row with the best F1 threshold."""
    return threshold_results.sort_values(by="f1", ascending=False).iloc[0]
