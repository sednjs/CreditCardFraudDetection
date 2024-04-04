#!pip install fastapi
#!pip install uvicorn
#!pip install pydantic

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Union
import joblib
import pandas as pd

import logging

logger = logging.getLogger("my_logger")
logging.basicConfig(level=logging.INFO)

# Use logger.info("message") instead of print("message")
logger.info("This is an info message")


# Load the trained model and preprocessors
model = joblib.load("decision_tree_model.joblib")
scaler = joblib.load("scaler.joblib")
woe_encoder = joblib.load("woe_encoder.joblib")

app = FastAPI(title="CreditCardFraudDetectionApp",
              description="CreditCardFraudDetectionApp allows you to predict if a transaction is fraudulent or not.")

class TransactionInput(BaseModel):
    merchant: str
    category: str
    amt: float
    last: str
    gender: int
    lat: float
    long: float
    city_pop: int
    job: str
    merch_lat: float
    merch_long: float

class PredictOutput(BaseModel):
    is_fraud: bool
    probability: object

@app.get("/")
def root():
    return "Hello! This is the credit card fraud prediction ML service!"

@app.post("/predict/", response_model=PredictOutput)
def predict_fraud(transaction: TransactionInput):
    # Convert the transaction input into a DataFrame
    transaction_df = pd.DataFrame([transaction.dict()])

    # Apply WOE encoding to categorical features
    encoded_df = woe_encoder.transform(transaction_df[['merchant', 'category', 'last', 'job']])
    print("Shape after encoding:", encoded_df.shape)  # Debug print

    # Ensure the encoded features replace the original ones in the DataFrame
    transaction_df.drop(['merchant', 'category', 'last', 'job'], axis=1, inplace=True)
    transaction_df = pd.concat([transaction_df, encoded_df], axis=1)

    # Ensure 'gender' is included correctly after preprocessing adjustments
    transaction_df['gender'] = transaction_df['gender'].astype(int)

    # Check if all expected features are present
    assert transaction_df.shape[1] == 11, f"Expected 11 features, got {transaction_df.shape[1]}"

    # Apply StandardScaler to all features
    scaled_features = scaler.transform(transaction_df)
    print("Shape after scaling:", scaled_features.shape)  # Debug print

    # Predict using the processed features
    prediction = model.predict(scaled_features)
    probability = model.predict_proba(scaled_features)[:, 1][0]
    return PredictOutput(is_fraud=bool(prediction[0]), probability=probability)

@app.get("/transactions/{transaction_id}")
def read_transaction(transaction_id: int, q: Union[str, None] = None):
    return {"transaction_id": transaction_id, "q": q}

@app.put("/transactions/{transaction_id}")
def update_transaction(transaction_id: int, transaction: TransactionInput):
    return {"transaction_amt": transaction.amt, "transaction_id": transaction_id}
