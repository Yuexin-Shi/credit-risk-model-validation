# Model Validation Report

## 1. Model Objective

This project validates a binary credit risk model that predicts whether a borrower will experience serious delinquency within two years. The model is intended for credit risk ranking, validation practice, and manual review prioritisation.

## 2. Data Quality Findings

The project uses the Give Me Some Credit dataset. Raw data is not committed to the repository and must be downloaded separately from Kaggle.

Key data quality treatments include:

- Age values equal to 0 are treated as missing.
- Delinquency values 96 and 98 are treated as anomalous and replaced with missing values.
- Missing values are handled using median imputation.
- Missing-value indicators are added during preprocessing.

## 3. Train/Test Split Design

The data is split using stratified sampling to preserve the target default rate across training, validation, and test sets. This is important because serious delinquency cases represent a small minority of borrowers.

## 4. Model Performance

The best-performing model is Histogram Gradient Boosting.

Key test metrics:

- ROC-AUC: 0.868
- Average precision: 0.405
- Brier score: 0.049

These results indicate strong ranking performance and reasonable probability calibration for credit risk prioritisation.

## 5. Calibration Assessment

The model uses probability calibration to improve the reliability of predicted default probabilities. Calibration is assessed using Brier score and calibration curve analysis.

A test Brier score of 0.049 suggests that the predicted probabilities are reasonably calibrated for portfolio-level risk ranking.

## 6. Threshold Strategy

A manual review threshold is selected based on a 10% review-rate strategy.

Selected probability cutoff:

- Threshold: 0.176

Borrowers with predicted default probability above this threshold are treated as high risk and recommended for manual review.

## 7. Risk Decile Analysis

Risk decile analysis is used to check whether borrowers with higher predicted probabilities also show higher observed default rates.

The model shows effective risk ranking if the top risk deciles have substantially higher observed default rates than the portfolio average.

## 8. PSI / Drift Analysis

Population Stability Index is used to compare score distributions between training and test samples.

- PSI: 0.0006

This indicates low score distribution drift in the validation sample.

## 9. SHAP Explainability

SHAP analysis identifies the main drivers of predicted default risk.

Important drivers include:

- Historical delinquency variables
- Revolving credit utilization
- Debt ratio
- Monthly income
- Age

These drivers are consistent with credit risk intuition and support model interpretability.

## 10. Limitations and Monitoring Recommendations

This project is for model validation practice and portfolio-level risk analysis. It should not be used as a sole automated credit decisioning system without additional governance.

Recommended monitoring:

- Track ROC-AUC, PR-AUC, Brier score, and calibration over time.
- Monitor feature and score drift using PSI.
- Reassess thresholds periodically.
- Add fairness testing before any production use.
