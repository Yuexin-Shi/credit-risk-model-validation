import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)


def probability_metrics(dataset_name, y_true, y_probability):
    return {
        "dataset": dataset_name,
        "roc_auc": roc_auc_score(y_true, y_probability),
        "average_precision": average_precision_score(y_true, y_probability),
        "brier_score": brier_score_loss(y_true, y_probability),
        "default_rate": y_true.mean(),
    }


def risk_decile_summary(y_true, y_probability):
    results = pd.DataFrame({
        "actual_result": y_true.values,
        "predicted_probability": y_probability,
    })

    results["risk_decile"] = (
        pd.qcut(
            results["predicted_probability"],
            q=10,
            labels=False,
            duplicates="drop",
        ) + 1
    )

    summary = results.groupby("risk_decile").agg(
        customer_count=("actual_result", "count"),
        default_count=("actual_result", "sum"),
        observed_default_rate=("actual_result", "mean"),
        average_predicted_probability=("predicted_probability", "mean"),
    ).reset_index()

    summary["observed_default_rate"] = summary["observed_default_rate"] * 100
    summary["average_predicted_probability"] = (
        summary["average_predicted_probability"] * 100
    )

    return summary


def threshold_performance(y_true, y_probability, threshold):
    y_pred = (y_probability >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    total = tn + fp + fn + tp

    return {
        "threshold": threshold,
        "actual_review_rate": (tp + fp) / total,
        "precision_default_rate": tp / (tp + fp) if (tp + fp) > 0 else np.nan,
        "recall_capture_rate": tp / (tp + fn) if (tp + fn) > 0 else np.nan,
        "false_positive_rate": fp / (fp + tn) if (fp + tn) > 0 else np.nan,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
    }


def calculate_psi(expected, actual, buckets=10):
    breakpoints = np.quantile(expected, np.linspace(0, 1, buckets + 1))
    breakpoints = np.unique(breakpoints)

    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    expected_counts, _ = np.histogram(expected, bins=breakpoints)
    actual_counts, _ = np.histogram(actual, bins=breakpoints)

    expected_pct = expected_counts / len(expected)
    actual_pct = actual_counts / len(actual)

    expected_pct = np.where(expected_pct == 0, 0.0001, expected_pct)
    actual_pct = np.where(actual_pct == 0, 0.0001, actual_pct)

    psi_values = (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)

    psi_table = pd.DataFrame({
        "bucket": range(1, len(psi_values) + 1),
        "expected_pct": expected_pct,
        "actual_pct": actual_pct,
        "psi": psi_values,
    })

    return psi_values.sum(), psi_table
