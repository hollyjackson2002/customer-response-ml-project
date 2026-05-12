"""
01_eda.py
Exploratory Data Analysis for Customer Response Prediction
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
RAW_DATA = ROOT / "data" / "raw" / "marketing_campaign.csv"
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)

df = pd.read_csv(RAW_DATA, sep="\t")

print("Dataset shape:", df.shape)
print("\nFirst five rows:")
print(df.head())
print("\nDataset information:")
print(df.info())
print("\nMissing values:")
print(df.isnull().sum())
print("\nDuplicate rows:", df.duplicated().sum())
print("\nDescriptive statistics:")
print(df.describe())

# Basic cleaning for EDA
df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], errors="coerce", dayfirst=True)
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

# 1. Response distribution
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="Response")
plt.title("Marketing Campaign Response Distribution")
plt.xlabel("Response: 0 = No, 1 = Yes")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.savefig(FIGURES / "01_response_distribution.png", dpi=300)
plt.close()

# 2. Income distribution
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="Income", kde=True)
plt.title("Customer Income Distribution")
plt.xlabel("Income")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(FIGURES / "02_income_distribution.png", dpi=300)
plt.close()

# 3. Age distribution
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="Age", kde=True)
plt.title("Customer Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(FIGURES / "03_age_distribution.png", dpi=300)
plt.close()

# 4. Total spending vs response
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="Response", y="Total_Spending")
plt.title("Total Spending by Campaign Response")
plt.xlabel("Response")
plt.ylabel("Total Spending")
plt.tight_layout()
plt.savefig(FIGURES / "04_total_spending_vs_response.png", dpi=300)
plt.close()

# 5. Income vs response
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="Response", y="Income")
plt.title("Income by Campaign Response")
plt.xlabel("Response")
plt.ylabel("Income")
plt.tight_layout()
plt.savefig(FIGURES / "05_income_vs_response.png", dpi=300)
plt.close()

# 6. Education vs response
plt.figure(figsize=(10, 5))
sns.countplot(data=df, x="Education", hue="Response")
plt.title("Campaign Response by Education Level")
plt.xlabel("Education")
plt.ylabel("Number of Customers")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(FIGURES / "06_education_vs_response.png", dpi=300)
plt.close()

# 7. Correlation heatmap
numeric_df = df.select_dtypes(include=["int64", "float64"])
plt.figure(figsize=(15, 11))
sns.heatmap(numeric_df.corr(), cmap="coolwarm", center=0)
plt.title("Correlation Heatmap of Numerical Features")
plt.tight_layout()
plt.savefig(FIGURES / "07_correlation_heatmap.png", dpi=300)
plt.close()

print("\nEDA complete. Figures saved in the figures folder.")
