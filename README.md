# Student Fee Defaulter Risk Prediction System

> An end-to-end machine learning pipeline for early identification of students at risk of fee default — covering data exploration, feature engineering, model comparison, SHAP explainability, and cost-sensitive threshold analysis.

---

## Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Target Engineering](#target-engineering)
- [Project Workflow](#project-workflow)
- [Model Performance](#model-performance)
- [Explainability Results](#explainability-results)
- [Threshold Analysis](#threshold-analysis)
- [Repository Structure](#repository-structure)
- [Key Outputs](#key-outputs)
- [Tools and Libraries](#tools-and-libraries)
- [How to Reproduce](#how-to-reproduce)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [Author](#author)

---

## Overview

Educational institutions often need early-warning systems to flag students at risk of financial difficulty before problems become irreversible. In many real-world settings, direct fee-payment records are unavailable, incomplete, or too delayed to act on.

This project addresses that gap by building a **fee-default risk prediction system** from student academic and contextual data. Since the dataset does not include an observed fee-default label, a **proxy target** was engineered from academic distress indicators — low final grades, repeated course failures, and high absenteeism.

The project goes beyond standard model training to emphasise:

- structured, reproducible experimentation
- leakage-aware dataset design (full vs. strict feature sets)
- interpretable predictions via SHAP
- practical threshold selection using a cost-weighted framework

---

## Problem Statement

Schools and finance administrators often cannot identify financially vulnerable students early enough for timely intervention. Without a systematic approach, at-risk students may only be detected after default has already occurred — too late for preventive action such as payment plan offers, bursaries, or academic support.

This project builds a machine learning pipeline that assigns a **fee-default risk score** to each student using features such as:

- academic performance (grades, failures)
- attendance history
- parental education and occupation
- family structure and support systems
- access to study resources

---

## Dataset

| Property | Details |
|----------|---------|
| Source | [Kaggle — Student Performance Dataset](https://www.kaggle.com/datasets/uciml/student-performance-data-set) |
| Records | 395 students |
| Original features | 33 columns |
| Domain | Portuguese secondary school (Mathematics cohort) |

Key variable groups include demographics (`sex`, `age`, `address`), family background (`famsize`, `Pstatus`, `Medu`, `Fedu`, `Mjob`, `Fjob`), support systems (`schoolsup`, `famsup`, `paid`, `internet`), and academic outcomes (`absences`, `failures`, `G1`, `G2`, `G3`).

---

## Target Engineering

> **Important:** This dataset contains no real fee-default column. The target variable is a constructed proxy.

A binary column `default` was engineered using academic distress signals:

```python
default = 1  if  (G3 < 10)  OR  (failures >= 2)  OR  (absences > 15)
          0  otherwise
```

To improve methodological rigour and transparency, two modelling datasets were created:

| Dataset | Description |
|---------|-------------|
| **Full** | All engineered predictors included |
| **Strict** | `G3`, `failures`, and `absences` excluded from predictors to reduce direct leakage from target construction |

This dual-dataset design makes the project more transparent and reflects how a real practitioner would handle label construction uncertainty.

---

## Project Workflow

### 1. Exploratory Data Analysis
- Inspected data structure, column types, and missing values
- Selected relevant features and validated class balance
- Confirmed variable relationships with the proxy target

### 2. Feature Engineering
- Binary encoding for binary categorical variables
- One-hot encoding for parental occupation columns (`Mjob`, `Fjob`)
- Derived composite features:

| Feature | Description |
|---------|-------------|
| `parent_edu_total` | Combined parental education score |
| `parent_edu_gap` | Absolute difference between parental education levels |
| `grade_avg_12` | Average of G1 and G2 |
| `grade_delta_12` | Change in grade from G1 to G2 |
| `support_total` | Total academic support indicators |
| `resource_support_total` | Access to paid tutoring and internet |
| `access_risk_score` | Composite resource access risk |
| `family_stability_score` | Composite family context score |

### 3. Model Experiments
Two classifiers were trained and compared on both datasets:

- Logistic Regression (baseline)
- Random Forest (champion)

### 4. Model Explainability
SHAP was applied for:
- Global feature importance (bar and beeswarm plots)
- Local explanation of individual student risk scores

### 5. Threshold Analysis
Multiple classification thresholds were evaluated across precision, recall, F1-score, and estimated intervention volume to determine the most operationally appropriate cutoff.

---

## Model Performance

### Comparison Table

| Dataset | Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---------|-------|----------|-----------|--------|-----|---------|
| Full | Random Forest | 0.9494 | 0.9655 | 0.9032 | 0.9333 | 0.9960 |
| Full | Logistic Regression | 0.9114 | 1.0000 | 0.7742 | 0.8727 | 0.9597 |
| Strict | Random Forest | 0.8354 | 0.8750 | 0.6774 | 0.7636 | 0.9066 |
| Strict | Logistic Regression | 0.8481 | 0.8800 | 0.7097 | 0.7857 | 0.8891 |

### Champion Model

**Random Forest — Full Feature Set**

Selected for the strongest ROC-AUC, F1-score, precision, and recall. The strict-dataset results are also reported to give an honest view of performance when target-correlated features are removed.

---

## Explainability Results

SHAP analysis identified the top predictive features as:

1. `G3` — final grade
2. `G2` — second-period grade
3. `grade_avg_12` — engineered grade average
4. `absences` — total absences
5. `G1` — first-period grade
6. `failures` — number of past course failures

### Key Patterns

The model consistently associated:
- lower grades → higher predicted risk
- more absences → higher predicted risk
- repeated failures → higher predicted risk

### Interpretation Note

Because `G3`, `failures`, and `absences` directly inform the engineered target, their high SHAP importance is expected. These results demonstrate that the model correctly learned the proxy-risk definition — not necessarily real-world causal factors. The strict-dataset results provide a fairer signal of what the model can learn from contextual features alone.

---

## Threshold Analysis

The default threshold of `0.50` was not accepted automatically. Multiple cutoffs were evaluated to balance precision, recall, and intervention volume.

### Results Summary

| Threshold | Accuracy | Precision | Recall | F1 |
|-----------|----------|-----------|--------|----|
| 0.20 | 0.9747 | 0.9394 | 1.0000 | 0.9688 |
| 0.30 | 0.9620 | 0.9286 | 0.9677 | 0.9478 |
| 0.50 | 0.9494 | 0.9655 | 0.9032 | 0.9333 |

### Recommendation

- **Threshold 0.20** — best for maximum recall; flags all high-risk students with high precision. Recommended when early intervention resources are readily available.
- **Threshold 0.30** — strong balance between recall and intervention volume. Recommended when capacity for outreach is more limited.

---

## Repository Structure

```text
fee-defaulter-prediction/
├── data/
│   ├── raw/                        # original, unmodified source data
│   ├── processed/                  # cleaned and feature-engineered datasets
│   ├── external/                   # third-party enrichment data (if any)
│   └── data_dictionary.md          # column definitions, types, and sources
│
├── notebooks/
│   ├── 01_eda.ipynb                # exploratory data analysis
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_experiments.ipynb  # baseline and advanced model comparison
│   ├── 04_model_explainability.ipynb
│   └── 05_threshold_analysis.ipynb
│
├── src/
│   ├── data/                       # ingestion, validation, splitting
│   ├── features/                   # build_features.py, transformers.py
│   ├── models/                     # train.py, predict.py, evaluate.py
│   ├── api/                        # FastAPI app and Pydantic schemas
│   └── utils/                      # logger, helpers
│
├── models/
│   ├── best_model.pkl              # serialised champion model
│   └── model_card.md              # intended use, limitations, fairness notes
│
├── reports/
│   ├── figures/                    # ROC curves, SHAP plots, confusion matrices
│   ├── model_results.csv
│   ├── shap_feature_importance.csv
│   └── threshold_results.csv
│
├── tests/                          # unit and integration tests
├── config.yaml                     # central pipeline configuration
├── requirements.txt
└── README.md
```

---

## Key Outputs

| Output File | Description |
|-------------|-------------|
| `data/processed/01_eda_output.csv` | Cleaned, filtered dataset post-EDA |
| `data/processed/02_features_full.csv` | Full feature matrix |
| `data/processed/02_features_strict.csv` | Strict (leakage-reduced) feature matrix |
| `reports/model_results.csv` | Model comparison metrics |
| `reports/shap_feature_importance.csv` | Global SHAP rankings |
| `reports/threshold_results.csv` | Threshold sweep results |
| `models/best_model.pkl` | Serialised Random Forest model |

---

## Tools and Libraries

| Category | Libraries |
|----------|-----------|
| Data manipulation | `pandas`, `numpy` |
| Visualisation | `matplotlib`, `seaborn` |
| Modelling | `scikit-learn` |
| Explainability | `shap` |
| Model persistence | `joblib` |
| Data access | `kagglehub` |
| Environment | Google Colab |

---

## How to Reproduce

### 1. Clone the repository

```bash
git clone https://github.com/Code-blize/fee-defaulter-prediction.git
cd fee-defaulter-prediction
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run notebooks in order

```
01_eda.ipynb
02_feature_engineering.ipynb
03_model_experiments.ipynb
04_model_explainability.ipynb
05_threshold_analysis.ipynb
```

Each notebook saves its outputs to `data/processed/` or `reports/` for the next stage to consume.

---

## Limitations

| Limitation | Detail |
|------------|--------|
| Proxy target | The `default` label is engineered, not observed from real payment records |
| Small dataset | 395 students — results may not generalise without additional data |
| Full model leakage | The full-feature model includes variables directly tied to target construction; strict-dataset results give a more conservative estimate |
| Single institution | Data originates from one Portuguese secondary school; generalisation requires revalidation |
| No macroeconomic signals | Inflation, employment, and household income data are not included |

---

## Future Improvements

- [ ] Train on real institutional fee-payment records with temporal splits
- [ ] Expand model comparison to XGBoost and LightGBM
- [ ] Add Optuna-based hyperparameter optimisation
- [ ] Build a Streamlit or FastAPI app for live risk scoring
- [ ] Write unit tests for the full pipeline
- [ ] Add a fairness audit across demographic subgroups
- [ ] Publish a full Model Card with intended use and ethical considerations

---

## Author

**Blessing Obasi-Uzoma**  
Physics Graduate (FUTO, B.Tech 2025) · Geospatial Data Scientist · AI/ML Practitioner

- GitHub: [Code-blize](https://github.com/Code-blize)
- LinkedIn: [Blessing Obasi-Uzoma](https://linkedin.com/in/blessing-obasi-uzoma)

---

## Acknowledgement

This project was developed as part of a hands-on machine learning learning journey focused on building practical, explainable, and professionally structured data science projects.

---

*Built with purpose · Designed for clarity · Grounded in real-world impact*
