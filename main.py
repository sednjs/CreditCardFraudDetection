#!pip install fastapi
#!pip install uvicorn
#!pip install pydantic

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel
from typing import Union
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from category_encoders import WOEEncoder
from datetime import datetime
import asyncio
import uvicorn

# Load the trained model
best_pipeline=joblib.load("best_pipeline.joblib")

api_title ="CreditCardFraudDetectionApp"
api_description="""
CreditCardFraudDetectionApp allows you to predict if a transaction is fraudulent or not.
"""
app=FastAPI(title=api_title,description=api_description)

class TransactionInput(BaseModel):
    merchant: str
    category: str
    amt: float
    last: str
    gender: str  
    lat: float
    long: float
    city_pop: int
    job: str
    merch_lat: float
    merch_long: float 
    hour: int
    month: int

class PredictOutput(BaseModel):
    is_fraud: bool
    probability: float   

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "body": exc.body}
    )

@app.get("/")
def root():
    # Ensure JSON structure is returned
    return {"msg": "Hello! This is the credit card fraud prediction ML service!"}

# FastAPI endpoint for prediction
@app.post("/predict", response_model=PredictOutput)
def predict_fraud(transaction: TransactionInput):
    data = {
        'merchant': [transaction.merchant],
        'category': [transaction.category],
        'amt': [transaction.amt],
        'last': [transaction.last],
        'gender': [transaction.gender],
        'lat': [transaction.lat],
        'long': [transaction.long],
        'city_pop': [transaction.city_pop],
        'job': [transaction.job],
        'merch_lat': [transaction.merch_lat],
        'merch_long': [transaction.merch_long],
        'hour': [transaction.hour],
        'month': [transaction.month]
    }
    input_data = pd.DataFrame(data)
    
    # input_data_processed=preprocessing_pipeline.transform(input_data)

    # predict
    prediction=best_pipeline.predict(input_data)
    probability=best_pipeline.predict_proba(input_data)[:,1]

    if prediction.size > 0:
        prediction_scalar = bool(prediction[0])
    else:
        prediction_scalar = False  # Default or fallback value if prediction is empty

    if probability.size > 0:
        probability_scalar = float(probability[0])
    else:
        probability_scalar = 0.0  # Default or fallback value if probability is empty

    return PredictOutput(is_fraud=prediction_scalar, probability=probability_scalar)

# Run FastAPI app
if __name__ =="__main copy__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", post=8000)
