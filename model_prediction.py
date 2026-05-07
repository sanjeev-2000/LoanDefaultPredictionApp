import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score
)

from xgboost import XGBClassifier


# =========================================================
# LOAD DATASET
# =========================================================
df = pd.read_csv("Loan_Default.csv")

# df['status'] = 1 - df['status']

def convert_age(age_value):

    # If already numeric
    try:
        return float(age_value)
    except:
        pass

    # If range format like 25-34
    if '-' in str(age_value):
        start, end = str(age_value).split('-')
        return (int(start) + int(end)) / 2

    return None


df['age'] = df['age'].apply(convert_age)


# =========================================================
# FEATURE ENGINEERING
# =========================================================

# Loan to income ratio
df['loan_income_ratio'] = (
    df['loan_amount'] / (df['income'] + 1)
)


# =========================================================
# FEATURES & TARGET
# =========================================================

# IMPORTANT:
# Replace 'status' with your actual target column name
#
# Expected:
# 1 = Default
# 0 = No Default
#
# If reversed in dataset:
# df['status'] = 1 - df['status']

X = df[[
    'age',
    'income',
    'loan_amount',
    'credit_score',
    'loan_income_ratio'
]]

y = df['status']


# =========================================================
# NUMERIC FEATURES
# =========================================================
numeric_features = [
    'age',
    'income',
    'loan_amount',
    'credit_score',
    'loan_income_ratio'
]


# =========================================================
# NUMERIC TRANSFORMER
# =========================================================
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])


# =========================================================
# PREPROCESSOR
# =========================================================
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features)
    ]
)


# =========================================================
# MODEL PIPELINE
# =========================================================
model = Pipeline(steps=[

    ('preprocessor', preprocessor),

    ('classifier', XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        gamma=1,
        scale_pos_weight=3,
        eval_metric='logloss',
        random_state=42
    ))
])


# =========================================================
# TRAIN TEST SPLIT
# =========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# =========================================================
# TRAIN MODEL
# =========================================================
model.fit(X_train, y_train)


# =========================================================
# PREDICTIONS
# =========================================================
y_pred = model.predict(X_test)

y_prob = model.predict_proba(X_test)[:, 1]


# =========================================================
# EVALUATION
# =========================================================
print("\n============================")
print("MODEL PERFORMANCE")
print("============================")

print(
    "\nAccuracy:",
    round(accuracy_score(y_test, y_pred), 4)
)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print(
    "\nROC-AUC Score:",
    round(roc_auc_score(y_test, y_prob), 4)
)


# =========================================================
# SAVE MODEL
# =========================================================
joblib.dump(model, "loan_default_model.pkl")

print("\nModel saved successfully!")


# =========================================================
# SAMPLE PREDICTION
# =========================================================
sample_data = pd.DataFrame({
    'age': [21],
    'income': [18000],
    'loan_amount': [450000],
    'credit_score': [350],
    'loan_income_ratio': [450000 / 18000]
})

probabilities = model.predict_proba(sample_data)[0]

default_probability = probabilities[0]
no_default_probability = probabilities[1]

if no_default_probability >= default_probability:
    prediction = "No Default"
    probability = no_default_probability
else:
    prediction = "Default"
    probability = default_probability

print({
    "prediction": prediction,
    "probability": round(float(probability), 2)
})