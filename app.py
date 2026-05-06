from fastapi import FastAPI
from pydantic import BaseModel

import pandas as pd
import joblib


# ==========================================
# LOAD MODEL
# ==========================================
model = joblib.load("loan_model.pkl")


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
    loan_amount: float
    income: float
    credit_score: float
    age: str


# ==========================================
# AGE CONVERTER
# ==========================================
def convert_age(age_range):
    start, end = age_range.split('-')
    return (int(start) + int(end)) / 2


# ==========================================
# PREDICTION ENDPOINT
# ==========================================
@app.post("/predict")
def predict(request: LoanRequest):

    # Feature Engineering
    loan_income_ratio = (
        request.loan_amount / (request.income + 1)
    )

    age_numeric = convert_age(request.age)

    # Create dataframe
    input_data = pd.DataFrame({
        'loan_amount': [request.loan_amount],
        'income': [request.income],
        'credit_score': [request.credit_score],
        'loan_income_ratio': [loan_income_ratio],
        'age_numeric': [age_numeric]
    })

    # Predict probability
    probability = model.predict_proba(input_data)[0][1]

    # Custom threshold
    threshold = 0.4

    prediction = (
        "Default"
        if probability >= threshold
        else "No Default"
    )

    return {
        "prediction": prediction,
        "default_probability": round(
            float(probability) * 100,
            2
        )
    }