from __future__ import annotations

from pathlib import Path
import pandas as pd


REQUIRED_RAW_COLUMNS = [
    "school", "sex", "age", "address", "famsize", "Pstatus", "Medu", "Fedu",
    "Mjob", "Fjob", "reason", "guardian", "traveltime", "studytime", "failures",
    "schoolsup", "famsup", "paid", "activities", "nursery", "higher", "internet",
    "romantic", "famrel", "freetime", "goout", "Dalc", "Walc", "health",
    "absences", "G1", "G2", "G3"
]


def validate_required_columns(df: pd.DataFrame, required_columns: list[str]) -> None:
    """Raise an error if any required columns are missing."""
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def validate_no_missing_values(df: pd.DataFrame) -> None:
    """Raise an error if the DataFrame contains missing values."""
    missing_counts = df.isnull().sum()
    missing_cols = missing_counts[missing_counts > 0]
    if not missing_cols.empty:
        raise ValueError(f"Missing values found:\n{missing_cols}")


def validate_target_column(df: pd.DataFrame, target_col: str = "default") -> None:
    """Validate target column existence and binary values."""
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")

    unique_values = sorted(df[target_col].dropna().unique().tolist())
    if unique_values not in ([0, 1], [0], [1]):
        raise ValueError(f"Target column '{target_col}' must be binary. Found: {unique_values}")


def validate_dataframe(df: pd.DataFrame, required_columns: list[str] | None = None) -> None:
    """Run core validation checks on a DataFrame."""
    if required_columns is not None:
        validate_required_columns(df, required_columns)
    validate_no_missing_values(df)
