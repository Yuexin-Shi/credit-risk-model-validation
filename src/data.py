from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def resolve_data_path():
    local_path = Path("../data/raw/cs-training.csv")
    colab_path = Path("/content/cs-training.csv")

    if local_path.exists():
        return local_path
    if colab_path.exists():
        return colab_path

    raise FileNotFoundError(
        "Cannot find cs-training.csv. Place it at ../data/raw/cs-training.csv "
        "or upload it to /content/cs-training.csv in Google Colab."
    )


def load_credit_data(data_path=None):
    if data_path is None:
        data_path = resolve_data_path()

    df = pd.read_csv(data_path)
    df = df.rename(columns={"Unnamed: 0": "customer_id"})
    return df


def clean_credit_data(df):
    df = df.copy()

    df.loc[df["age"] == 0, "age"] = np.nan

    delinquency_columns = [
        "NumberOfTime30-59DaysPastDueNotWorse",
        "NumberOfTime60-89DaysPastDueNotWorse",
        "NumberOfTimes90DaysLate",
    ]

    for column in delinquency_columns:
        df[column] = df[column].replace([96, 98], np.nan)

    return df


def split_credit_data(df, target="SeriousDlqin2yrs", test_size=0.20, random_state=42):
    X = df.drop(columns=["customer_id", target])
    y = df[target].astype(int)

    return train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )
