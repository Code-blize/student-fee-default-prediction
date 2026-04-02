from pathlib import Path
import pandas as pd

from src.features.build_features import filter_relevant_columns, create_default_target


def test_filter_relevant_columns_returns_expected_columns():
    df = pd.DataFrame({
        "sex": ["F"],
        "age": [18],
        "address": ["U"],
        "famsize": ["GT3"],
        "Pstatus": ["T"],
        "Medu": [4],
        "Fedu": [4],
        "Mjob": ["teacher"],
        "Fjob": ["services"],
        "traveltime": [2],
        "studytime": [2],
        "failures": [0],
        "schoolsup": ["no"],
        "famsup": ["yes"],
        "paid": ["yes"],
        "higher": ["yes"],
        "internet": ["yes"],
        "famrel": [4],
        "absences": [3],
        "G1": [14],
        "G2": [15],
        "G3": [15],
        "extra_col": ["ignore"]
    })

    filtered = filter_relevant_columns(df)
    assert "extra_col" not in filtered.columns
    assert filtered.shape[1] == 22


def test_create_default_target_binary():
    df = pd.DataFrame({
        "sex": ["F", "M"],
        "age": [18, 17],
        "address": ["U", "R"],
        "famsize": ["GT3", "LE3"],
        "Pstatus": ["T", "A"],
        "Medu": [4, 1],
        "Fedu": [4, 1],
        "Mjob": ["teacher", "other"],
        "Fjob": ["services", "other"],
        "traveltime": [2, 3],
        "studytime": [2, 1],
        "failures": [0, 2],
        "schoolsup": ["no", "yes"],
        "famsup": ["yes", "no"],
        "paid": ["yes", "no"],
        "higher": ["yes", "no"],
        "internet": ["yes", "no"],
        "famrel": [4, 3],
        "absences": [3, 20],
        "G1": [14, 7],
        "G2": [15, 6],
        "G3": [15, 8],
    })

    out = create_default_target(df)
    assert "default" in out.columns
    assert set(out["default"].unique()) <= {0, 1}
