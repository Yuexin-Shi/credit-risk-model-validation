# Credit Risk Model Validation

An end-to-end credit risk modelling and model validation project using the Give Me Some Credit dataset.

## Project Objectives

This project aims to:

- Build credit default prediction models
- Handle missing values and anomalous observations
- Compare logistic regression and tree-based models
- Evaluate model discrimination and calibration
- Analyse feature importance and model explainability
- Detect data drift and potential model risk

## Dataset

The project uses the Give Me Some Credit dataset.

The raw data is not included in this repository. It can be downloaded separately from Kaggle.

## Project Workflow

- [x] Data quality audit
- [x] Exploratory data analysis
- [x] Data preprocessing
- [x] Logistic regression baseline
- [x] Tree-based model comparison
- [x] Model calibration
- [x] SHAP explainability analysis
- [x] Data drift analysis
- [x] Model card
- [x] API deployment

## Key Results

- Built an end-to-end credit risk modeling pipeline using the Give Me Some Credit dataset.
- Compared logistic regression and tree-based models for default prediction.
- Evaluated model performance using ROC-AUC, PR-AUC, KS statistic, precision, recall, and calibration metrics.
- Conducted threshold analysis to support risk-based approval, rejection, and manual review decisions.
- Identified important borrower-level risk factors and potential model risk issues.
  
## Key Metrics

The models will be evaluated using:

- ROC-AUC
- PR-AUC
- KS statistic
- Precision and Recall
- Brier Score
- Calibration Curve

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- XGBoost
- SHAP

## Repository Structure

```text
credit-risk-model-validation/
├── data/
├── notebooks/
├── src/
├── reports/
├── README.md
├── requirements.txt
└── .gitignore
