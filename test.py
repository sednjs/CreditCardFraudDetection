#!pip install fastapi
#!pip install uvicorn
#!pip install pydantic

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Union
import joblib
import numpy as np

import requests

# API endpoint URL
url = 'http://127.0.0.1:8000/predict'

# data to request
transaction_data = {
    "amt": 2.86,
    "category": 'personal_care',
    "merch_lat": 33.986391,
    "merch_long": -81.200714
}

# POST
response = requests.post(url, json=transaction_data)

# response
if response.status_code == 200:
    prediction_result = response.json()
    print("API response:", prediction_result)
else:
    print("API request failed:", response.status_code, response.text)

