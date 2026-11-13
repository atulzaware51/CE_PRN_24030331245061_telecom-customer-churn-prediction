Set-Location "C:\Users\Asus\Documents\Customer_Churn_ML"

@'
.venv/
venv/
__pycache__/
*.py[cod]
.pytest_cache/
.mypy_cache/
.ruff_cache/
.ipynb_checkpoints/
.env
.env.*
!.env.example
.streamlit/secrets.toml
.DS_Store
Thumbs.db
*.log
*.tmp
'@ | Set-Content ".gitignore"

@'
# Telecom Customer Churn Prediction and Retention Recommendation System Using Machine Learning

## 1. Project Overview

This project uses machine learning to predict whether a telecom customer is likely to churn or stay. It includes a Streamlit web application for customer predictions, model comparison, churn risk analysis and project information.

## 2. Problem Statement

Customer churn affects telecom businesses through lost revenue and customer relationships. Identifying customers who may leave can help businesses investigate risks and plan appropriate retention activities.

## 3. Objectives

- Analyze telecom customer data.
- Prepare data for machine learning.
- Train and evaluate classification models.
- Predict customer churn.
- Compare multiple machine learning models.
- Present predictions and analysis through a Streamlit application.
- Support retention planning using model predictions and customer information.

## 4. Technology Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib
- Streamlit
- Jupyter Notebook

## 5. Dataset

Dataset file: `data/telecom_customer_churn.csv`

The original dataset contains 7,043 customer records and 38 columns.

The target column is `Customer Status`. The modeling workflow excludes customers whose status is `Joined`, leaving 6,589 records for churn/stay modeling.

Target classes:
- Stayed
- Churned

## 6. Machine Learning Models

The project contains three saved machine learning pipelines:

1. Logistic Regression
2. Support Vector Machine (SVM)
3. K-Nearest Neighbors (KNN)

Saved artifacts:
- `models/logistic_regression_pipeline.pkl`
- `models/svm_pipeline.pkl`
- `models/knn_pipeline.pkl`
- `models/app_config.json`

## 7. Model Evaluation

Previously recorded evaluation results:

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 83.84% | 71.93% | 70.59% | 71.26% | 90.59% |
| SVM | 84.14% | 74.48% | 67.11% | 70.60% | 90.30% |
| KNN | 81.87% | 69.01% | 65.51% | 67.22% | 85.61% |

These are previously recorded evaluation results. Re-run the evaluation workflow to independently reproduce them.

Confusion matrix and comparison reports are stored in `reports/`.

## 8. Application

Main application file:

`app/streamlit_app.py`

Existing application sections:
- Customer Prediction
- Model Comparison Dashboard
- Customer Churn Risk Analysis
- About Project

The separate three-model comparison module is located at:

`app/features/model_consensus/page.py`

Its integration into the main application should be verified before claiming it is an available navigation page.

## 9. Project Structure

```text
Customer_Churn_ML/
|
|-- app/
|   |-- streamlit_app.py
|   |-- features/
|       |-- model_consensus/
|           |-- page.py
|
|-- data/
|   |-- telecom_customer_churn.csv
|
|-- models/
|   |-- logistic_regression_pipeline.pkl
|   |-- svm_pipeline.pkl
|   |-- knn_pipeline.pkl
|   |-- app_config.json
|
|-- notebooks/
|   |-- 01_dataset_inspection.ipynb
|   |-- 02_preprocessing.ipynb
|
|-- reports/
|   |-- model_comparison.csv
|   |-- logistic_regression_confusion_matrix.csv
|   |-- svm_confusion_matrix.csv
|   |-- knn_confusion_matrix.csv
|
|-- src/
|   |-- inspect_data.py
|
|-- requirements.txt
|-- README.md
|-- .gitignore