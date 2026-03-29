# Model Evaluation Report

## Overview

This report summarizes the experimental performance of the machine learning models developed for the **Student Fee Defaulter Risk Prediction System**.

The project used a proxy target for fee-default risk because the dataset did not contain an observed fee-payment default label.

Two dataset variants were evaluated:

- **Full dataset**: includes all engineered predictors
- **Strict dataset**: excludes `G3`, `failures`, and `absences` to reduce direct leakage from target construction

Two models were compared:

- Logistic Regression
- Random Forest

---

## Evaluation Metrics

The following metrics were used:

- **Accuracy**: overall proportion of correct predictions
- **Precision**: proportion of predicted high-risk students who were actually high-risk
- **Recall**: proportion of true high-risk students that were correctly identified
- **F1-score**: harmonic mean of precision and recall
- **ROC-AUC**: ability to rank higher-risk students above lower-risk students

Because this project is framed as an early-warning system, **recall** is especially important.

---

## Model Comparison Results

| Dataset | Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|--------|-------|----------|-----------|--------|----------|---------|
| Full   | Random Forest        | 0.9494 | 0.9655 | 0.9032 | 0.9333 | 0.9960 |
| Full   | Logistic Regression  | 0.9114 | 1.0000 | 0.7742 | 0.8727 | 0.9597 |
| Strict | Random Forest        | 0.8354 | 0.8750 | 0.6774 | 0.7636 | 0.9066 |
| Strict | Logistic Regression  | 0.8481 | 0.8800 | 0.7097 | 0.7857 | 0.8891 |

---

## Champion Model

The best-performing model was:

**Random Forest on the full dataset**

### Why it was selected
It achieved the best overall results across the key metrics:

- highest ROC-AUC
- highest F1-score
- strong precision
- strong recall

This suggests the model can separate higher-risk and lower-risk students very effectively within the project setting.

---

## Interpretation of Full vs Strict Results

The full dataset performed noticeably better than the strict dataset.

This is expected because the full dataset includes:

- `G3`
- `failures`
- `absences`

These variables were also used in constructing the proxy target. As a result, they contain strong predictive signal.

However, the strict dataset still produced meaningful results:

- Random Forest ROC-AUC: 0.9066
- Logistic Regression ROC-AUC: 0.8891

This is an important finding because it shows that the project retains predictive value even after removing the most direct target-construction features.

---

## Explainability Summary

SHAP analysis showed that the model relied mostly on:

- `G3`
- `G2`
- `grade_avg_12`
- `absences`
- `G1`
- `failures`

These variables were the strongest contributors to model predictions.

In general:

- lower grades increased predicted risk
- higher absenteeism increased predicted risk
- repeated failures increased predicted risk

This behavior is consistent with the target-engineering logic.

---

## Threshold Analysis

The default classification threshold of 0.50 was evaluated against alternative thresholds.

The strongest threshold by F1-score was:

**0.20**

### Metrics at threshold 0.20
- Accuracy: 0.9747
- Precision: 0.9394
- Recall: 1.0000
- F1-score: 0.9688
- Predicted positives: 33

This threshold identified all high-risk students in the test set while maintaining high precision.

### Practical threshold interpretation
- **0.20** is appropriate when recall is the priority
- **0.30** is a good alternative when intervention capacity is more limited

---

## Key Findings

1. Random Forest outperformed Logistic Regression on the full dataset.
2. The strict dataset provided a useful leakage-aware benchmark.
3. SHAP confirmed that academic distress variables were the dominant drivers of predictions.
4. Threshold tuning improved the practical decision quality of the model.
5. The project demonstrates a full machine learning workflow, not just model fitting.

---

## Limitations

This evaluation should be interpreted with the following limitations in mind:

- the target is engineered, not observed from real fee-payment outcomes
- the dataset is small
- performance on the full dataset may be optimistic due to target-related predictors
- results are best understood as a strong portfolio demonstration of risk modeling methodology

---

## Conclusion

The Student Fee Defaulter Risk Prediction System successfully demonstrated an end-to-end workflow for building, evaluating, explaining, and tuning a classification model for early-risk detection.

The selected Random Forest model showed strong predictive performance, and the threshold analysis provided a practical cutoff for risk screening.

Although the target is a proxy rather than a real fee-default label, the project demonstrates a thoughtful and professionally structured approach to machine learning problem framing, evaluation, and interpretability.
