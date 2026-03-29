from __future__ import annotations

import os
from pathlib import Path
import pandas as pd
import kagglehub


DEFAULT_DATASET_SLUG = "devansodariya/student-performance-data"


def download_dataset(dataset_slug: str = DEFAULT_DATASET_SLUG) -> Path:
    """Download dataset from KaggleHub and return the local path."""
    path = kagglehub.dataset_download(dataset_slug)
    return Path(path)


def list_dataset_files(dataset_path: str | Path) -> list[str]:
    """List files in the downloaded dataset directory."""
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")
    return sorted(os.listdir(dataset_path))


def find_csv_file(dataset_path: str | Path) -> Path:
    """Find the first CSV file in the dataset directory."""
    dataset_path = Path(dataset_path)
    csv_files = sorted([p for p in dataset_path.iterdir() if p.suffix.lower() == ".csv"])
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {dataset_path}")
    return csv_files[0]


def load_raw_dataset(dataset_path: str | Path) -> pd.DataFrame:
    """Load the raw CSV dataset into a DataFrame."""
    csv_path = find_csv_file(dataset_path)
    return pd.read_csv(csv_path)


def save_raw_copy(df: pd.DataFrame, output_dir: str | Path, filename: str = "raw_student_data.csv") -> Path:
    """Save a copy of the raw dataset."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / filename
    df.to_csv(output_path, index=False)
    return output_path
