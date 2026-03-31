from __future__ import annotations

from pathlib import Path
import pandas as pd


KEEP_COLS = [
    "sex", "age", "address", "famsize", "Pstatus",
    "Medu", "Fedu", "Mjob", "Fjob",
    "traveltime", "studytime", "failures",
    "schoolsup", "famsup", "paid",
    "higher", "internet", "famrel",
    "absences", "G1", "G2", "G3"
]

LEAKAGE_COLS = ["G3", "failures", "absences"]

BINARY_MAPS = {
    "sex": {"F": 0, "M": 1},
    "address": {"R": 0, "U": 1},
    "famsize": {"LE3": 0, "GT3": 1},
    "Pstatus": {"A": 0, "T": 1},
    "schoolsup": {"no": 0, "yes": 1},
    "famsup": {"no": 0, "yes": 1},
    "paid": {"no": 0, "yes": 1},
    "higher": {"no": 0, "yes": 1},
    "internet": {"no": 0, "yes": 1},
}


def filter_relevant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only columns relevant to the project."""
    missing = [col for col in KEEP_COLS if col not in df.columns]
    if missing:
        raise KeyError(f"Missing columns for filtering: {missing}")
    return df[KEEP_COLS].copy()


def create_default_target(
    df: pd.DataFrame,
    g3_threshold: int = 10,
    failure_threshold: int = 2,
    absence_threshold: int = 15
) -> pd.DataFrame:
    """
    Create proxy fee-default target.
    default = 1 when academic distress signals are present.
    """
    df = df.copy()

    required_cols = ["G3", "failures", "absences"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise KeyError(f"Missing columns for target creation: {missing}")

    df["default"] = (
        (df["G3"] < g3_threshold) |
        (df["failures"] >= failure_threshold) |
        (df["absences"] > absence_threshold)
    ).astype(int)

    return df


def encode_binary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map binary categorical columns to 0/1."""
    df = df.copy()

    for col, mapping in BINARY_MAPS.items():
        if col not in df.columns:
            raise KeyError(f"Expected column '{col}' not found.")
        df[col] = df[col].map(mapping)

    return df


def encode_job_columns(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode parental occupation columns."""
    df = df.copy()

    required_cols = ["Mjob", "Fjob"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise KeyError(f"Missing job columns for encoding: {missing}")

    df = pd.get_dummies(df, columns=required_cols, drop_first=True)

    bool_cols = df.select_dtypes(include="bool").columns
    if len(bool_cols) > 0:
        df[bool_cols] = df[bool_cols].astype(int)

    return df


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create engineered features used in modeling."""
    df = df.copy()

    required_cols = [
        "Medu", "Fedu", "G1", "G2", "traveltime", "internet",
        "Pstatus", "famrel", "schoolsup", "famsup", "paid"
    ]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise KeyError(f"Missing columns for feature engineering: {missing}")

    df["parent_edu_total"] = df["Medu"] + df["Fedu"]
    df["parent_edu_gap"] = (df["Medu"] - df["Fedu"]).abs()

    df["grade_avg_12"] = (df["G1"] + df["G2"]) / 2
    df["grade_delta_12"] = df["G2"] - df["G1"]

    df["support_total"] = df["schoolsup"] + df["famsup"]
    df["resource_support_total"] = df["schoolsup"] + df["famsup"] + df["paid"]

    df["access_risk_score"] = df["traveltime"] + (1 - df["internet"])
    df["family_stability_score"] = df["Pstatus"] + df["famrel"]

    return df


def build_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full feature pipeline from filtered dataset to engineered table."""
    df = encode_binary_columns(df)
    df = encode_job_columns(df)
    df = add_engineered_features(df)
    return df


def split_full_and_strict(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return full and leakage-aware strict datasets."""
    missing = [col for col in LEAKAGE_COLS if col not in df.columns]
    if missing:
        raise KeyError(f"Leakage columns not found: {missing}")

    df_full = df.copy()
    df_strict = df.drop(columns=LEAKAGE_COLS).copy()
    return df_full, df_strict


def save_dataframe(df: pd.DataFrame, output_path: str | Path) -> Path:
    """Save a single DataFrame."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


def save_feature_tables(
    df_full: pd.DataFrame,
    df_strict: pd.DataFrame,
    output_dir: str | Path
) -> tuple[Path, Path]:
    """Save full and strict feature tables."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    full_path = output_dir / "02_features_full.csv"
    strict_path = output_dir / "02_features_strict.csv"

    df_full.to_csv(full_path, index=False)
    df_strict.to_csv(strict_path, index=False)

    return full_path, strict_path
