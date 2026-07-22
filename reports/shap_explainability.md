# SHAP Explainability Report

## Objective

This report summarizes the SHAP explainability analysis for the selected credit risk model. SHAP values are used to explain which borrower-level variables contribute most to predicted default risk.

## Model Explained

The SHAP analysis explains the Histogram Gradient Boosting model used in the validation workflow.

## Methodology

The model was trained using cleaned borrower-level credit risk features. Missing values were imputed using median imputation with missing-value indicators. SHAP values were calculated on a sample of the processed test set to explain model behavior.

## Key Drivers of Predicted Default Risk

The SHAP analysis indicates that the following variables are important drivers of predicted default risk:

- Revolving utilization of unsecured lines
- Number of times 90 days late
- Number of 30-59 days past due events
- Number of 60-89 days past due events
- Debt ratio
- Monthly income
- Age

## Interpretation

Higher recent delinquency counts and higher revolving credit utilization generally increase predicted default risk. Income, debt burden, and borrower age also contribute to model risk ranking.

## Model Risk Notes

- SHAP values explain model behavior, not causal relationships.
- Correlated credit variables may share explanatory contribution.
- SHAP analysis should be reviewed together with calibration, threshold performance, and stability metrics.
- Explanations should support model validation and business review, not replace formal governance.

## Reference Notebook

See `notebooks/07_shap_explainability.ipynb` for the full analysis workflow.
