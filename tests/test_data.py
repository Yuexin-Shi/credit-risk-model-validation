import pandas as pd

from src.data import clean_credit_data, split_credit_data


def make_sample_data():
    return pd.DataFrame({
        "customer_id": [1, 2, 3, 4, 5, 6],
        "SeriousDlqin2yrs": [0, 1, 0, 1, 0, 1],
        "RevolvingUtilizationOfUnsecuredLines": [0.1, 0.8, 0.2, 0.9, 0.3, 0.7],
        "age": [45, 0, 38, 52, 61, 29],
        "NumberOfTime30-59DaysPastDueNotWorse": [0, 1, 96, 2, 0, 98],
        "DebtRatio": [0.2, 0.5, 0.3, 0.7, 0.1, 0.6],
        "MonthlyIncome": [5000, 3000, None, 7000, 9000, 2500],
        "NumberOfOpenCreditLinesAndLoans": [5, 4, 3, 8, 6, 2],
        "NumberOfTimes90DaysLate": [0, 1, 96, 2, 0, 98],
        "NumberRealEstateLoansOrLines": [1, 0, 0, 2, 1, 0],
        "NumberOfTime60-89DaysPastDueNotWorse": [0, 1, 96, 2, 0, 98],
        "NumberOfDependents": [1, 0, 2, 1, None, 0],
    })


def test_clean_credit_data_replaces_known_anomalies():
    df = make_sample_data()

    cleaned = clean_credit_data(df)

    assert cleaned.loc[1, "age"] != 0
    assert cleaned["age"].isna().sum() == 1
    assert cleaned["NumberOfTimes90DaysLate"].isna().sum() == 2
    assert cleaned["NumberOfTime30-59DaysPastDueNotWorse"].isna().sum() == 2
    assert cleaned["NumberOfTime60-89DaysPastDueNotWorse"].isna().sum() == 2


def test_split_credit_data_returns_train_and_test_sets():
    df = clean_credit_data(make_sample_data())

    X_train, X_test, y_train, y_test = split_credit_data(
        df,
        test_size=0.5,
        random_state=42,
    )

    assert "SeriousDlqin2yrs" not in X_train.columns
    assert "customer_id" not in X_train.columns
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
