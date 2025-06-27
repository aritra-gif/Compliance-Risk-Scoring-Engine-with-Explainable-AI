# Compliance Risk Scoring Engine (AI/ML)

This project simulates a machine learning-based compliance risk scoring system using synthetic escalation and resolution data. Built with Python and Scikit-learn, this system predicts high-risk cases to support governance teams in identifying compliance anomalies.

## 🔍 Features

- Stacked ensemble model (Random Forest, Gradient Boosting, Logistic Regression)
- SHAP Explainability for audit readiness
- Synthetic dataset simulating compliance patterns

## 📊 Outputs

- Classification metrics
- ROC AUC score
- SHAP summary plot for top contributing features

## 📦 Libraries

- pandas
- numpy
- scikit-learn
- shap
- matplotlib

## ▶️ Usage

```bash
pip install -r requirements.txt
python compliance_risk_scoring.py
```

## 📁 File Structure

- `compliance_risk_scoring.py` - Main script
- `README.md` - Project documentation
