# from fastapi import FastAPI

# app = FastAPI()

# @app.get("/")
# async def root():
#     return {"message": "Credit Card Fraud Detection"}

# import os

# current_directory = os.getcwd()
# print("Current Directory:", current_directory)

# import sys
# import subprocess
# subprocess.check_call([sys.executable, "-m", "pip", "install", "requests"])


from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from sklearn.preprocessing import StandardScaler
from category_encoders import WOEEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

app = FastAPI()

# Define the input data model
class Transaction(BaseModel):
    amt: float
    city_pop: int
    job: str
    merchant: str
    category: str
    gender: str
    lat: float
    long: float
    month: int
    hour: int

# Load the pre-trained model and scaler
model = joblib.load("random_forest_model.pkl")
scaler = joblib.load("scaler.pkl")

# Define the prediction function
def predict_fraud(transaction: Transaction):
    # Convert the transaction data into a DataFrame
    data = pd.DataFrame([transaction.dict()])
    
    # Encoding
    data['merchant'] = data['merchant'].apply(lambda x: x.replace('fraud_', ''))
    data['gender'] = data['gender'].map({'F': 0, 'M': 1})
    for col in ['job', 'merchant', 'category', 'lat', 'long']:
        data[col] = WOEEncoder().fit_transform(data[col], data['amt'])
    
    # Scaling
    data_scaled = scaler.transform(data)
    
    # Make predictions
    prediction = model.predict(data_scaled)[0]
    probability = model.predict_proba(data_scaled)[0][1]  # Probability of fraud
    
    return {"fraud_prediction": bool(prediction), "fraud_probability": probability}

# Endpoint to make predictions
@app.post("/predict")
async def predict(transaction: Transaction):
    return predict_fraud(transaction)

# Root endpoint
@app.get("/")
async def root():
    return {"message": "Credit Card Fraud Detection"}



## start app in terminal
## uvicorn main:app --reload



# # sending POST requests


# import requests

# # Define the transaction data
# transaction_data = {
#     "amt": 100.0,
#     "city_pop": 10000,
#     "job": "manager",
#     "merchant": "store",
#     "category": "grocery",
#     "gender": "M",
#     "lat": 40.7128,
#     "long": -74.0060,
#     "month": 6,
#     "hour": 15
# }

# # Send a POST request to the /predict endpoint
# response = requests.post("http://localhost:8000/predict", json=transaction_data)

# # Print the response
# print(response.json())