"""
02_preprocessing.py
Data Cleaning, Feature Engineering and Train-Test Split
"""

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

ROOT = Path(__file__).resolve().parents[1]
RAW_DATA = ROOT / "data" / "raw" / "marketing_campaign.csv"
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"

PROCESSED.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)

df = pd.read_csv(RAW_DATA, sep="\t")

# Remove rows with missing Income values
df = df.dropna(subset=["Income"]).copy()

# Convert date column
df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], errors="coerce", dayfirst=True)

# Feature engineering
df["Age"] = 2026 - df["Year_Birth"]

spending_columns = [
    "MntWines", "MntFruits", "MntMeatProducts",
    "MntFishProducts", "MntSweetProducts", "MntGoldProds"
]
df["Total_Spending"] = df[spending_columns].sum(axis=1)

purchase_columns = [
    "NumDealsPurchases", "NumWebPurchases",
    "NumCatalogPurchases", "NumStorePurchases"
]
df["Total_Purchases"] = df[purchase_columns].sum(axis=1)

df["Children"] = df["Kidhome"] + df["Teenhome"]

# Remove columns that are identifiers, dates, or constants
columns_to_drop = [
    "ID",
    "Dt_Customer",
    "Z_CostContact",
    "Z_Revenue"
]

df = df.drop(columns=columns_to_drop, errors="ignore")

# Encode categorical variables
categorical_columns = df.select_dtypes(include=["object"]).columns.tolist()
label_encoders = {}

for column in categorical_columns:
    encoder = LabelEncoder()
    df[column] = encoder.fit_transform(df[column].astype(str))
    label_encoders[column] = encoder

joblib.dump(label_encoders, MODELS / "label_encoders.joblib")

# Define features and target
X = df.drop("Response", axis=1)
y = df["Response"]

# Train-test split with stratification because Response is likely imbalanced
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Scale data for Logistic Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

joblib.dump(scaler, MODELS / "standard_scaler.joblib")

# Save processed data
X_train.to_csv(PROCESSED / "X_train.csv", index=False)
X_test.to_csv(PROCESSED / "X_test.csv", index=False)
y_train.to_csv(PROCESSED / "y_train.csv", index=False)
y_test.to_csv(PROCESSED / "y_test.csv", index=False)

pd.DataFrame(X_train_scaled, columns=X_train.columns).to_csv(
    PROCESSED / "X_train_scaled.csv", index=False
)
pd.DataFrame(X_test_scaled, columns=X_test.columns).to_csv(
    PROCESSED / "X_test_scaled.csv", index=False
)

print("Preprocessing complete.")
print("Training features:", X_train.shape)
print("Testing features:", X_test.shape)
print("Files saved in data/processed.")
