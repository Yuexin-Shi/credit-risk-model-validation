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

## Planned Workflow

- [x] Data quality audit
- [ ] Exploratory data analysis
- [ ] Data preprocessing
- [ ] Logistic regression baseline
- [ ] Tree-based model comparison
- [ ] Model calibration
- [ ] SHAP explainability analysis
- [ ] Data drift analysis
- [ ] Model card
- [ ] API deployment

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
