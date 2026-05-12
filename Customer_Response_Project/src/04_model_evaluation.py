"""
04_model_evaluation.py
Evaluate Models and Save Results, Confusion Matrices, ROC Curves and Feature Importance
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)

X_test = pd.read_csv(PROCESSED / "X_test.csv")
X_test_scaled = pd.read_csv(PROCESSED / "X_test_scaled.csv")
y_test = pd.read_csv(PROCESSED / "y_test.csv").squeeze()

logistic_model = joblib.load(MODELS / "logistic_regression.joblib")
random_forest_model = joblib.load(MODELS / "random_forest.joblib")

with open(MODELS / "advanced_model_name.txt") as f:
    advanced_model_name = f.read().strip()

advanced_model = joblib.load(MODELS / f"{advanced_model_name}.joblib")

models = {
    "Logistic Regression": (logistic_model, X_test_scaled),
    "Random Forest": (random_forest_model, X_test),
    advanced_model_name.replace("_", " ").title(): (advanced_model, X_test)
}

results = []

for model_name, (model, test_data) in models.items():
    y_pred = model.predict(test_data)

    if hasattr(model, "predict_proba"):
        y_probability = model.predict_proba(test_data)[:, 1]
    else:
        y_probability = y_pred

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_probability)

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC": roc_auc
    })

    print("\n" + "=" * 60)
    print(model_name)
    print("=" * 60)
    print(classification_report(y_test, y_pred, zero_division=0))

    # Confusion matrix
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    plt.savefig(FIGURES / f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png", dpi=300)
    plt.close()

    # ROC curve
    RocCurveDisplay.from_predictions(y_test, y_probability)
    plt.title(f"ROC Curve - {model_name}")
    plt.tight_layout()
    plt.savefig(FIGURES / f"roc_curve_{model_name.lower().replace(' ', '_')}.png", dpi=300)
    plt.close()

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="F1 Score", ascending=False)
results_df.to_csv(PROCESSED / "model_results.csv", index=False)

print("\nModel comparison:")
print(results_df)

# Feature importance from Random Forest
feature_importance = pd.DataFrame({
    "Feature": X_test.columns,
    "Importance": random_forest_model.feature_importances_
}).sort_values(by="Importance", ascending=False)

feature_importance.to_csv(PROCESSED / "feature_importance.csv", index=False)

plt.figure(figsize=(10, 6))
sns.barplot(
    data=feature_importance.head(10),
    x="Importance",
    y="Feature"
)
plt.title("Top 10 Most Important Features - Random Forest")
plt.tight_layout()
plt.savefig(FIGURES / "feature_importance_random_forest.png", dpi=300)
plt.close()

print("\nEvaluation complete.")
print("Results saved to data/processed/model_results.csv.")
print("Figures saved in the figures folder.")
