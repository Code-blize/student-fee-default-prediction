# Project Overview

## Title
Student Fee Defaulter Prediction System

## Summary
This project is an end-to-end machine learning system for predicting student fee-default risk using academic, family, and support-related indicators.

It was built to demonstrate a complete machine learning workflow, including:

- exploratory data analysis
- feature engineering
- model experimentation
- explainability with SHAP
- threshold analysis
- FastAPI deployment

## Problem Context
Educational institutions may need to identify students who are at higher risk of fee default early enough for intervention. In practice, direct fee-payment labels may not always be available.

To address this, the project uses an academic dataset and engineers a proxy target called `default` based on academic distress signals such as:

- low final grade
- repeated failures
- high absenteeism

## Core Deliverables
- cleaned and processed feature tables
- trained classification models
- explainability outputs
- threshold tuning results
- deployed FastAPI inference endpoint

## Best Model
The best-performing model was a Random Forest classifier trained on the full feature set.

## Deployment
The project is deployed with FastAPI on Render and exposes endpoints for:

- health check
- API documentation
- prediction
