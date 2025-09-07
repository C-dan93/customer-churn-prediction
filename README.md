# Customer Churn Prediction with Explainable AI

End-to-end machine learning pipeline predicting telecommunications customer churn with SHAP explainability and production cloud deployment.

## Live Demo
- **API:** https://churn-prediction-api-fgqu7sreyq-uc.a.run.app
- **Documentation:** https://churn-prediction-api-fgqu7sreyq-uc.a.run.app/docs

## Results Summary
- **Accuracy:** 77.4% with 78.6% recall
- **Model:** Logistic Regression with RFE feature selection  
- **Top Factor:** Tenure (SHAP importance: 0.166)
- **Business Impact:** Identifies 4 out of 5 customers who will churn

## Project Overview

This project implements a complete customer churn prediction solution including:
- Comprehensive exploratory data analysis
- Systematic model comparison and feature selection
- SHAP explainability for business insights
- Production-ready REST API with cloud deployment

## Technical Stack
- **ML:** scikit-learn, XGBoost, SHAP
- **API:** FastAPI, Docker
- **Cloud:** Google Cloud Run
- **Data:** Pandas, NumPy, Matplotlib, Seaborn

## Key Findings
- New customers (0-12 months) show 47.7% churn rate vs 9.5% for established customers
- Month-to-month contracts have 42.7% churn vs 2.8% for two-year contracts
- Fiber optic customers require special retention attention
- Electronic check payment method increases churn risk

## Repository Structure
- `notebooks/`: Jupyter notebooks with EDA and model development
- `src/`: Production API code
- `models/`: Trained model artifacts
- `docs/`: Technical documentation and reports

## Quick Start
```bash
# Test the live API
curl https://churn-prediction-api-fgqu7sreyq-uc.a.run.app/health

# Or visit the interactive docs
https://churn-prediction-api-fgqu7sreyq-uc.a.run.app/docs
