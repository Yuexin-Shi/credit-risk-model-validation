import argparse
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline

from src.data import clean_credit_data, load_credit_data
from src.evaluation import probability_metrics, threshold_performance
from src.features import build_tree_preprocessor


DEFAULT_DATA_PATH = Path("data/raw/cs-training.csv")
DEFAULT_MODEL_PATH = Path("models/credit_risk_model.pkl")
DEFAULT_METADATA_PATH = Path("models/model_metadata.json")
DEFAULT_REVIEW_RATE = 0.10


def build_model(random_state=42):
    base_model = HistGradientBoostingClassifier(
        learning_rate=0.06,
        max_iter=250,
        l2_regularization=0.01,
        random_state=random_state,
    )

    try:
        calibrated_model = CalibratedClassifierCV(
            estimator=base_model,
            method="isotonic",
            cv=3,
        )
    except TypeError:
        calibrated_model = CalibratedClassifierCV(
            base_estimator=base_model,
            method="isotonic",
            cv=3,
        )

    return Pipeline([
        ("preprocessor", build_tree_preprocessor()),
        ("model", calibrated_model),
    ])


def choose_threshold(y_probability, review_rate=DEFAULT_REVIEW_RATE):
    return float(np.quantile(y_probability, 1 - review_rate))


def train_model(data_path, model_path, metadata_path, random_state=42):
    df = clean_credit_data(load_credit_data(data_path))
    X = df.drop(columns=["customer_id", "SeriousDlqin2yrs"])
    y = df["SeriousDlqin2yrs"].astype(int)

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=random_state,
    )

    model = build_model(random_state=random_state)
    model.fit(X_train, y_train)

    test_probability = model.predict_proba(X_test)[:, 1]
    threshold = choose_threshold(test_probability)

    metrics = probability_metrics("test", y_test, test_probability)
    threshold_metrics = threshold_performance(y_test, test_probability, threshold)

    artifact = {
        "model": model,
        "feature_columns": list(X.columns),
        "threshold": threshold,
        "risk_bands": {
            "low": 0.05,
            "medium": threshold,
        },
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    with model_path.open("wb") as file:
        pickle.dump(artifact, file)

    metadata = {
        "model_type": "Calibrated HistGradientBoostingClassifier",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "data_path": str(data_path),
        "model_path": str(model_path),
        "review_rate": DEFAULT_REVIEW_RATE,
        "threshold": threshold,
        "metrics": metrics,
        "threshold_metrics": threshold_metrics,
        "feature_columns": list(X.columns),
    }

    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the final credit risk model and save model artifacts."
    )
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--metadata-path", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    metadata = train_model(
        data_path=args.data_path,
        model_path=args.model_path,
        metadata_path=args.metadata_path,
        random_state=args.random_state,
    )

    print("Training complete.")
    print(f"Model artifact: {metadata['model_path']}")
    print(f"Metadata: {args.metadata_path}")
    print(json.dumps(metadata["metrics"], indent=2))


if __name__ == "__main__":
    main()
