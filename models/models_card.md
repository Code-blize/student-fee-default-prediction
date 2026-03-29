# Model Card: Student Fee Defaulter Risk Prediction

## Model Details

**Model name:** Random Forest Classifier  
**Task:** Binary classification  
**Prediction target:** Fee-default risk proxy (`default`)  
**Champion model selected from:** Model comparison experiments on full and strict feature sets

This model predicts whether a student is at **higher** or **lower** risk of fee default based on academic, family, and support-related indicators.

---

## Intended Use

This model is intended for:

- early-risk screening
- institutional intervention planning
- academic and financial support prioritization
- educational analytics demonstrations and portfolio use

The model is designed to help identify students who may benefit from **earlier review or intervention**.

---

## Out-of-Scope Use

This model should **not** be used for:

- direct financial decision-making without human review
- punitive action against students
- real billing enforcement
- scholarship denial or admission rejection
- any fully automated high-stakes decision

This project is a **risk-screening prototype**, not a production-ready financial system.

---

## Dataset

**Source:** Kaggle Student Performance Dataset  
**Records:** 395 students  
**Original features:** 33

The dataset contains student demographic, academic, and family-related variables such as:

- age
- parental education
- parental occupation
- family support
- internet access
- attendance
- grades (`G1`, `G2`, `G3`)

---

## Target Construction

The dataset does not contain an observed school-fee default label.

A proxy target called `default` was engineered using academic distress indicators:

- low final grade (`G3`)
- repeated failures (`failures`)
- high absenteeism (`absences`)

### Target definition
- `1` = higher fee-default risk
- `0` = lower fee-default risk

This means the model predicts a **fee-default risk proxy**, not real payment default.

---

## Feature Engineering

The following engineered features were created:

- `parent_edu_total`
- `parent_edu_gap`
- `grade_avg_12`
- `grade_delta_12`
- `support_total`
- `resource_support_total`
- `access_risk_score`
- `family_stability_score`

Binary categorical variables were encoded, and parental occupation columns were one-hot encoded.

---

## Model Training

Two dataset versions were evaluated:

### Full dataset
Includes all engineered predictors, including variables used in target construction.

### Strict dataset
Excludes:
- `G3`
- `failures`
- `absences`

This was done to reduce direct leakage from the proxy target definition.

Two models were compared:

- Logistic Regression
- Random Forest

The best-performing model was the **Random Forest trained on the full dataset**.

---

## Performance Summary

### Champion Model Performance

- **Accuracy:** 0.9494
- **Precision:** 0.9655
- **Recall:** 0.9032
- **F1-score:** 0.9333
- **ROC-AUC:** 0.9960

### Threshold Analysis
The default 0.50 threshold was not automatically used.

The best threshold by F1-score was:

**Threshold = 0.20**

Metrics at threshold 0.20:

- **Accuracy:** 0.9747
- **Precision:** 0.9394
- **Recall:** 1.0000
- **F1-score:** 0.9688

A threshold of **0.30** also performed strongly and may be more practical when intervention resources are limited.

---

## Explainability

SHAP analysis showed that the model relied mainly on:

- `G3`
- `G2`
- `grade_avg_12`
- `absences`
- `G1`
- `failures`

In general:

- lower grades increased predicted risk
- higher absenteeism increased predicted risk
- repeated failures increased predicted risk

---

## Limitations

This model has important limitations:

1. The target is engineered, not directly observed from real fee-payment data.
2. The dataset is relatively small.
3. The full model includes variables tied to target construction, which may inflate performance.
4. This is a portfolio-quality prototype, not a production financial-risk system.
5. The model should be interpreted as a screening tool, not a causal system.

---

## Ethical Considerations

This model should be used carefully.

Potential risks include:

- overinterpreting academic struggle as financial vulnerability
- unfair treatment of students without contextual review
- false positives that may cause unnecessary concern
- false negatives that may miss students needing support

Any real deployment should include:

- human review
- fairness auditing
- additional real financial/payment data
- transparent communication about limitations

---

## Recommendations for Real-World Improvement

To strengthen the model in future work:

- use real fee-payment history
- include temporal payment records
- validate on larger institutional datasets
- add fairness and bias analysis
- build a monitored deployment pipeline
- compare additional models such as XGBoost or LightGBM

---

## Author

**Blessing Obasi-Uzoma**  
Aspiring Data Scientist | Physics Background | Machine Learning and Analytics Projects
