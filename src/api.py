from typing import Dict

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI(
    title="Credit Risk Model API",
    description="Example API structure for credit risk model scoring.",
    version="0.1.0",
)


class CreditApplication(BaseModel):
    RevolvingUtilizationOfUnsecuredLines: float
    age: float
    NumberOfTime30_59DaysPastDueNotWorse: float
    DebtRatio: float
    MonthlyIncome: float | None = None
    NumberOfOpenCreditLinesAndLoans: float
    NumberOfTimes90DaysLate: float
    NumberRealEstateLoansOrLines: float
    NumberOfTime60_89DaysPastDueNotWorse: float
    NumberOfDependents: float | None = None


@app.get("/")
def health_check() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict(application: CreditApplication) -> Dict[str, object]:
    input_data = pd.DataFrame([application.model_dump()])

    return {
        "message": "API structure is ready. Connect a trained model artifact before production scoring.",
        "input_columns": list(input_data.columns),
        "default_probability": None,
        "risk_band": None,
    }
