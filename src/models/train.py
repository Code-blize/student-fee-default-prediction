from __future__ import annotations

from pathlib import Path
import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from src.models.evaluate import classification_metrics


TARGET_COL = "default"


def load_feature_table(csv_path: str | Path) -> pd.DataFrame:
    """Load a processed feature table."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Feature table not found: {csv_path}")
    return pd.read_csv(csv_path)


def split_X_y(df: pd.DataFrame, target_col: str = TARGET_COL):
    """Split dataframe into X and y."""
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found.")
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def stratified_split(X, y, test_size: float = 0.2, random_state: int = 42):
    """Perform a stratified train-test split."""
    return train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )


def get_models() -> dict:
    """Return model dictionary."""
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, random_state=42))
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            random_state=42
        ),
    }


def train_and_evaluate_one(model, X_train, X_test, y_train, y_test, dataset_name: str, model_name: str):
    """Train one model and return metrics and fitted estimator."""
    fitted_model = clone(model)
    fitted_model.fit(X_train, y_train)

    y_pred = fitted_model.predict(X_test)
    y_prob = fitted_model.predict_proba(X_test)[:, 1]

    metrics = classification_metrics(y_test, y_pred, y_prob)
    metrics["dataset"] = dataset_name
    metrics["model"] = model_name

    return metrics, fitted_model


def run_experiments(df_full: pd.DataFrame, df_strict: pd.DataFrame) -> tuple[pd.DataFrame, dict, dict]:
    """Run experiments on full and strict datasets."""
    X_full, y_full = split_X_y(df_full)
    X_strict, y_strict = split_X_y(df_strict)

    Xf_train, Xf_test, yf_train, yf_test = stratified_split(X_full, y_full)
    Xs_train, Xs_test, ys_train, ys_test = stratified_split(X_strict, y_strict)

    models = get_models()
    results = []
    trained_models = {}

    for model_name, model in models.items():
        metrics, fitted_model = train_and_evaluate_one(
            model, Xf_train, Xf_test, yf_train, yf_test, "full", model_name
        )
        results.append(metrics)
        trained_models[f"full_{model_name}"] = fitted_model

    for model_name, model in models.items():
        metrics, fitted_model = train_and_evaluate_one(
            model, Xs_train, Xs_test, ys_train, ys_test, "strict", model_name
        )
        results.append(metrics)
        trained_models[f"strict_{model_name}"] = fitted_model

    results_df = pd.DataFrame(results).sort_values(by=["roc_auc", "f1"], ascending=False)

    split_store = {
        "full": (Xf_train, Xf_test, yf_train, yf_test),
        "strict": (Xs_train, Xs_test, ys_train, ys_test),
    }

    return results_df, trained_models, split_store


def get_best_model(results_df: pd.DataFrame, trained_models: dict):
    """Return best row and matching fitted model."""
    best_row = results_df.iloc[0]
    model_key = f"{best_row['dataset']}_{best_row['model']}"
    best_model = trained_models[model_key]
    return best_row, best_model


def save_results(results_df: pd.DataFrame, output_path: str | Path) -> Path:
    """Save experiment results CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    return output_path


def save_model(model, output_path: str | Path) -> Path:
    """Save trained model artifact."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    return output_path


def save_best_split(best_dataset: str, split_store: dict, output_dir: str | Path) -> dict[str, Path]:
    """Save the train/test split used by the best model."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test = split_store[best_dataset]

    paths = {
        "X_train": output_dir / "03_best_X_train.csv",
        "X_test": output_dir / "03_best_X_test.csv",
        "y_train": output_dir / "03_best_y_train.csv",
        "y_test": output_dir / "03_best_y_test.csv",
    }

    X_train.to_csv(paths["X_train"], index=False)
    X_test.to_csv(paths["X_test"], index=False)
    y_train.to_csv(paths["y_train"], index=False)
    y_test.to_csv(paths["y_test"], index=False)

    return paths
