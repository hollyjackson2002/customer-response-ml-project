"""
03_model_training.py
Train Logistic Regression, Random Forest and XGBoost/Gradient Boosting Models
"""

from pathlib import Path
import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)

X_train = pd.read_csv(PROCESSED / "X_train.csv")
X_train_scaled = pd.read_csv(PROCESSED / "X_train_scaled.csv")
y_train = pd.read_csv(PROCESSED / "y_train.csv").squeeze()

# 1. Logistic Regression
logistic_model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
logistic_model.fit(X_train_scaled, y_train)
joblib.dump(logistic_model, MODELS / "logistic_regression.joblib")

# 2. Random Forest
random_forest_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    class_weight="balanced",
    random_state=42
)
random_forest_model.fit(X_train, y_train)
joblib.dump(random_forest_model, MODELS / "random_forest.joblib")

# 3. XGBoost if available; otherwise use Gradient Boosting
try:
    from xgboost import XGBClassifier

    advanced_model = XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42
    )
    advanced_model_name = "xgboost"
except ImportError:
    advanced_model = GradientBoostingClassifier(random_state=42)
    advanced_model_name = "gradient_boosting"

advanced_model.fit(X_train, y_train)
joblib.dump(advanced_model, MODELS / f"{advanced_model_name}.joblib")

with open(MODELS / "advanced_model_name.txt", "w") as f:
    f.write(advanced_model_name)

print("Model training complete.")
print("Saved models:")
print("- Logistic Regression")
print("- Random Forest")
print(f"- {advanced_model_name}")
