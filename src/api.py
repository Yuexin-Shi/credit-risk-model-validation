import pickle
from pathlib import Path
from typing import Dict

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field


app = FastAPI(
    title="Credit Risk Model API",
    description="Example API structure for credit risk model scoring.",
    version="0.1.0",
)

MODEL_PATH = Path("models/credit_risk_model.pkl")
MODEL_ARTIFACT = None


def load_model_artifact():
    if not MODEL_PATH.exists():
        return None

    with MODEL_PATH.open("rb") as file:
        return pickle.load(file)


@app.on_event("startup")
def startup_event():
    global MODEL_ARTIFACT
    MODEL_ARTIFACT = load_model_artifact()


class CreditApplication(BaseModel):
    RevolvingUtilizationOfUnsecuredLines: float
    age: float
    NumberOfTime30_59DaysPastDueNotWorse: float = Field(
        alias="NumberOfTime30-59DaysPastDueNotWorse"
    )
    DebtRatio: float
    MonthlyIncome: float | None = None
    NumberOfOpenCreditLinesAndLoans: float
    NumberOfTimes90DaysLate: float
    NumberRealEstateLoansOrLines: float
    NumberOfTime60_89DaysPastDueNotWorse: float = Field(
        alias="NumberOfTime60-89DaysPastDueNotWorse"
    )
    NumberOfDependents: float | None = None


@app.get("/")
def health_check() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict(application: CreditApplication) -> Dict[str, object]:
    if MODEL_ARTIFACT is None:
        return {
            "message": "Model artifact not found. Run `python -m src.train` first.",
            "default_probability": None,
            "risk_band": None,
        }

    input_data = pd.DataFrame([application.model_dump(by_alias=True)])
    input_data = input_data[MODEL_ARTIFACT["feature_columns"]]

    default_probability = float(MODEL_ARTIFACT["model"].predict_proba(input_data)[0, 1])
    threshold = MODEL_ARTIFACT["threshold"]

    if default_probability < MODEL_ARTIFACT["risk_bands"]["low"]:
        risk_band = "low"
    elif default_probability < threshold:
        risk_band = "medium"
    else:
        risk_band = "high"

    return {
        "default_probability": default_probability,
        "risk_band": risk_band,
        "manual_review_recommended": default_probability >= threshold,
        "threshold": threshold,
    }
