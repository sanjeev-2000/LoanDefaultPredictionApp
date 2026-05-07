from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import pandas as pd
import joblib


# ==========================================
# LOAD MODEL
# ==========================================
model = joblib.load("loan_default_model.pkl")


# ==========================================
# CREATE FASTAPI APP
# ==========================================
app = FastAPI(
    title="Loan Default Prediction API"
)


# ==========================================
# REQUEST BODY
# ==========================================
class LoanRequest(BaseModel):

    age: float = Field(
        ...,
        ge=18,
        le=100,
        description="Age must be between 18 and 100"
    )

    income: float = Field(
        ...,
        gt=0,
        description="Income must be greater than 0"
    )

    loan_amount: float = Field(
        ...,
        gt=0,
        description="Loan amount must be greater than 0"
    )

    credit_score: float = Field(
        ...,
        ge=300,
        le=850,
        description="Credit score must be between 300 and 850"
    )


# ==========================================
# PREDICTION ENDPOINT
# ==========================================
@app.post("/predict")
def predict(request: LoanRequest):

    # ==========================================
    # ADDITIONAL BUSINESS VALIDATIONS
    # ==========================================

    # Loan amount should not be absurdly high
    if request.loan_amount > 10000000:
        raise HTTPException(
            status_code=400,
            detail="Loan amount exceeds allowed limit"
        )

    # Income sanity check
    if request.income < 1000:
        raise HTTPException(
            status_code=400,
            detail="Income is unrealistically low"
        )

    # ==========================================
    # FEATURE ENGINEERING
    # ==========================================
    loan_income_ratio = (
        request.loan_amount / (request.income + 1)
    )

    # ==========================================
    # CREATE INPUT DATAFRAME
    # ==========================================
    input_data = pd.DataFrame({
        'age': [request.age],
        'income': [request.income],
        'loan_amount': [request.loan_amount],
        'credit_score': [request.credit_score],
        'loan_income_ratio': [loan_income_ratio]
    })

    # ==========================================
    # PREDICT PROBABILITIES
    # ==========================================
    probabilities = model.predict_proba(input_data)[0]

    # Class Mapping
    # 0 = Default
    # 1 = No Default
    default_probability = probabilities[0]
    no_default_probability = probabilities[1]

    # ==========================================
    # FINAL PREDICTION
    # ==========================================
    if no_default_probability >= default_probability:
        prediction = "No Default"
        probability = no_default_probability
    else:
        prediction = "Default"
        probability = default_probability

    # ==========================================
    # RESPONSE
    # ==========================================
    return {
        "prediction": prediction,
        "probability": round(float(probability), 2)
    }