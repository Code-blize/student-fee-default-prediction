import pandas as pd

from src.models.train import run_experiments, get_best_model


def make_df(n=30):
    rows = []
    for i in range(n):
        default = 1 if i % 3 == 0 else 0
        rows.append({
            "sex": i % 2,
            "age": 16 + (i % 3),
            "address": i % 2,
            "famsize": 1,
            "Pstatus": 1,
            "Medu": 2,
            "Fedu": 2,
            "traveltime": 2,
            "studytime": 2,
            "schoolsup": 0,
            "famsup": 1,
            "paid": 1,
            "higher": 1,
            "internet": 1,
            "famrel": 4,
            "G1": 8 if default else 14,
            "G2": 7 if default else 15,
            "G3": 6 if default else 15,
            "Mjob_other": 1,
            "Fjob_other": 1,
            "parent_edu_total": 4,
            "parent_edu_gap": 0,
            "grade_avg_12": 7.5 if default else 14.5,
            "grade_delta_12": -1 if default else 1,
            "support_total": 1,
            "resource_support_total": 2,
            "access_risk_score": 2,
            "family_stability_score": 5,
            "failures": 2 if default else 0,
            "absences": 20 if default else 3,
            "default": default,
        })
    return pd.DataFrame(rows)


def test_run_experiments():
    df_full = make_df()
    df_strict = df_full.drop(columns=["G3", "failures", "absences"])

    results_df, trained_models, split_store = run_experiments(df_full, df_strict)

    assert not results_df.empty
    assert "roc_auc" in results_df.columns
    assert len(trained_models) == 4
    assert "full" in split_store
    assert "strict" in split_store


def test_get_best_model():
    df_full = make_df()
    df_strict = df_full.drop(columns=["G3", "failures", "absences"])

    results_df, trained_models, _ = run_experiments(df_full, df_strict)
    best_row, best_model = get_best_model(results_df, trained_models)

    assert "dataset" in best_row.index
    assert best_model is not None
