# Credit Risk Model Card

## Model Overview

This project develops a binary classification model to predict serious delinquency within two years using the Give Me Some Credit dataset.

## Intended Use

The model is intended for credit risk analysis, model validation practice, and portfolio-level risk ranking. It should not be used as a sole automated credit decisioning system without additional governance, fairness review, and regulatory validation.

## Target Variable

- `SeriousDlqin2yrs`
- 1: borrower experienced serious delinquency within two years
- 0: borrower did not experience serious delinquency within two years

## Features

The model uses borrower-level financial and delinquency variables, including revolving utilization, age, debt ratio, monthly income, number of credit lines, real estate loans, dependents, and historical delinquency counts.

## Data Preparation

- Missing values are imputed using median imputation.
- Missing value indicators are added.
- Age value equal to 0 is treated as missing.
- Delinquency values 96 and 98 are treated as anomalous and replaced with missing values.
- Data is split into training, validation, and test sets using stratified sampling.

## Model

The final validation workflow uses a histogram gradient boosting classifier with probability calibration.

## Evaluation Metrics

The model is evaluated using:

- ROC-AUC
- Average precision / PR-AUC
- Brier score
- Calibration curve
- Threshold performance
- Risk decile analysis
- PSI stability analysis

## Key Risks and Limitations

- The dataset is historical and may not represent current lending populations.
- The model may contain bias if input variables correlate with protected characteristics.
- The model should be monitored for data drift and calibration drift.
- The project does not include production-grade monitoring, fairness testing, or regulatory approval.

## Monitoring Recommendations

- Track ROC-AUC, PR-AUC, Brier score, and calibration over time.
- Monitor feature distribution drift using PSI.
- Review approval / rejection thresholds periodically.
- Revalidate the model before production use.
