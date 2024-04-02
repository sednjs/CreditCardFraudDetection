#!pip install fastapi
#!pip install uvicorn
#!pip install pydantic

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Union
import joblib
import numpy as np

model=joblib.load("decision_tree_model.joblib")

# Version 1
api_title = "CreditCardFraudDetectionApp"
api_description = """
CreditCardFraudDetectionApp allows you to predict if a transaction is fraudulent or not.
"""

app = FastAPI(title=api_title, description=api_description)

class TransactionInput(BaseModel):
    amt: float
    category: str
    merch_lat: float
    merch_long: float

class PredictOutput(BaseModel):
    is_fraud: bool
    probability: float

@app.get("/")
def root():
    return "Hello! This is the credit card fraud prediction ML service!"

@app.post("/predict/", response_model=PredictOutput)
def predict_fraud(transaction: TransactionInput):
    try:
        features = np.array([[transaction.amt, transaction.merch_lat, transaction.merch_long]])
        prediction = model.predict(features)
        probability = model.predict_proba(features)[:, 1][0]
        return PredictOutput(is_fraud=bool(prediction[0]), probability=float(probability))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/transactions/{transaction_id}")
def read_transaction(transaction_id: int, q: Union[str, None] = None):
    return {"transaction_id": transaction_id, "q": q}

@app.put("/transactions/{transaction_id}")
def update_transaction(transaction_id: int, transaction: TransactionInput):
    return {"transaction_amt": transaction.amt, "transaction_id": transaction_id}

import asyncio

# This part is usually at the bottom of the script
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("my_fastapi_app:app", host="127.0.0.1", port=8000, reload=True)

