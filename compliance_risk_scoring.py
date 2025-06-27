
# compliance_risk_scoring.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
import shap
import matplotlib.pyplot as plt

# Step 1: Create synthetic compliance dataset
np.random.seed(42)
n = 1000
df = pd.DataFrame({
    "flag_count": np.random.poisson(2, n),
    "escalation_frequency": np.random.normal(5, 2, n),
    "response_time_avg": np.random.normal(48, 12, n),
    "policy_breach_score": np.random.uniform(0, 1, n),
    "repeat_offender": np.random.randint(0, 2, n),
    "resolution_time": np.random.normal(72, 20, n),
    "high_risk": np.random.binomial(1, 0.3, n)
})

X = df.drop("high_risk", axis=1)
y = df["high_risk"]

# Step 2: Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

# Step 3: Build Stacking Classifier
estimators = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42))
]
stack_model = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(),
    cv=5
)
stack_model.fit(X_train, y_train)

# Step 4: Evaluate
y_pred = stack_model.predict(X_test)
print(classification_report(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, stack_model.predict_proba(X_test)[:, 1]))

# Step 5: SHAP Explainability
explainer = shap.Explainer(stack_model.predict, X_test)
shap_values = explainer(X_test[:50])
shap.summary_plot(shap_values, X_test[:50], show=True)
