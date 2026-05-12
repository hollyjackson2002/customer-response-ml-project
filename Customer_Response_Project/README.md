# Customer Response Prediction Using Machine Learning

## Project Overview
This project predicts whether a customer will respond positively to a marketing campaign using demographic, income, purchasing and campaign history features.

The project supports the CSC-44112 Assessment Part 2 requirements by covering:
- Exploratory Data Analysis
- Data cleaning and preprocessing
- Machine learning model development
- Model evaluation and comparison
- Feature importance analysis
- Real-world business and ethical discussion

## Dataset
Dataset file used:

`data/raw/marketing_campaign.csv`

Target variable:

`Response`

- `1` = customer accepted/responded to the campaign
- `0` = customer did not respond

## Models Used
- Logistic Regression
- Random Forest
- XGBoost if installed, otherwise Gradient Boosting is used as a fallback

## Folder Structure

```text
Customer_Response_Project/
│
├── data/
│   ├── raw/
│   │   └── marketing_campaign.csv
│   └── processed/
│
├── figures/
├── models/
├── src/
│   ├── 01_eda.py
│   ├── 02_preprocessing.py
│   ├── 03_model_training.py
│   └── 04_model_evaluation.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

## How to Run the Project

Run the files in this order:

```bash
python src/01_eda.py
python src/02_preprocessing.py
python src/03_model_training.py
python src/04_model_evaluation.py
```

## Main Outputs

The scripts create:
- EDA graphs in the `figures/` folder
- processed training and testing data in `data/processed/`
- trained models in the `models/` folder
- model comparison results in `data/processed/model_results.csv`

## Report Use

Use the generated figures and result tables in the final report:
- Response distribution
- Income distribution
- Age distribution
- Spending vs response
- Correlation heatmap
- Confusion matrices
- ROC curves
- Feature importance chart
- Model comparison table

## Jupyter Notebooks

The `notebooks/` folder contains simple step-by-step notebooks that match the report sections:

1. `01_Exploratory_Data_Analysis.ipynb` — dataset overview, missing values, descriptive statistics and EDA graphs.
2. `02_Preprocessing_and_Feature_Engineering.ipynb` — data cleaning, encoding, scaling and train-test split.
3. `03_Model_Training.ipynb` — Logistic Regression, Random Forest and XGBoost model training.
4. `04_Results_and_Evaluation.ipynb` — evaluation metrics, confusion matrices, ROC curves, feature importance and learning curve.

Run the notebooks in numerical order.
