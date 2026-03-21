import os
import pandas as pd
import kagglehub


def download_dataset() -> str:
    path = kagglehub.dataset_download("devansodariya/student-performance-data")
    return path


def list_dataset_files(path: str) -> list[str]:
    return os.listdir(path)


def load_dataset(path: str) -> pd.DataFrame:
    files = os.listdir(path)
    csv_files = [f for f in files if f.endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError("No CSV file found in downloaded dataset.")
    file_path = os.path.join(path, csv_files[0])
    return pd.read_csv(file_path)
