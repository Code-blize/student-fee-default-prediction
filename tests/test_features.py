import pandas as pd

from src.features.build_features import (
    encode_binary_columns,
    encode_job_columns,
    add_engineered_features,
    build_feature_table,
)


def sample_df():
    return pd.DataFrame({
        "sex": ["F"],
        "age": [18],
        "address": ["U"],
        "famsize": ["GT3"],
        "Pstatus": ["T"],
        "Medu": [4],
        "Fedu": [3],
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
        "default": [0],
    })


def test_encode_binary_columns():
    df = encode_binary_columns(sample_df())
    assert df.loc[0, "sex"] in [0, 1]
    assert df.loc[0, "internet"] in [0, 1]


def test_encode_job_columns():
    df = encode_binary_columns(sample_df())
    df = encode_job_columns(df)
    assert any(col.startswith("Mjob_") for col in df.columns)
    assert any(col.startswith("Fjob_") for col in df.columns)


def test_add_engineered_features():
    df = encode_binary_columns(sample_df())
    df = encode_job_columns(df)
    df = add_engineered_features(df)

    expected = [
        "parent_edu_total",
        "parent_edu_gap",
        "grade_avg_12",
        "grade_delta_12",
        "support_total",
        "resource_support_total",
        "access_risk_score",
        "family_stability_score",
    ]

    for col in expected:
        assert col in df.columns


def test_build_feature_table():
    df = build_feature_table(sample_df())
    assert "grade_avg_12" in df.columns
    assert "resource_support_total" in df.columns
