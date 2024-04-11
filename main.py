#!pip install fastapi
#!pip install uvicorn
#!pip install pydantic

from fastapi import FastAPI, HTTPException
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


# Load the trained model and preprocessors
model, preprocessing_pipeline = joblib.load("decision_tree_model.joblib")

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

@app.get("/")
def root():
    return {"msg": "Hello! This is the credit card fraud prediction ML service!"}

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
    
    input_data_processed=preprocessing_pipeline.transform(input_data)

    # predict
    prediction=model.predict(input_data_processed)
    probability=model.predict_proba(input_data_processed)[:,1]

    return PredictOutput(is_fraud=bool(prediction),probability=float(probability))

