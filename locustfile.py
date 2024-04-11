# import sys
# import subprocess
# subprocess.check_call([sys.executable, "-m", "pip", "install", "locust"])
# import random

# from locust import HttpUser, constant_pacing, task


# class UnitPriceUser(HttpUser):
#     host = "http://127.0.0.1:8000"
#     wait_time = constant_pacing(1)

import random
from locust import HttpUser, task, between

class MyUser(HttpUser):
    host = "http://127.0.0.1:8000"  # Set the base URL of your FastAPI application
    wait_time = between(1, 5)  # Simulate a wait time between 1 to 5 seconds

    @task
    def predict_fraud(self):
        transaction_data = {
            "merchant": "fraud_Kirlin and Sons",
            "category": "personal_care",
            "amt": 2.86,
            "last": "Elliott",
            "gender": "M",
            "lat": 33.9659,
            "long": -80.9355,
            "city_pop": 333497,
            "job": "Mechanical engineer",
            "merch_lat": 33.986391,
            "merch_long": -81.200714,
            "hour": 3,
            "month": 7
        }

        response = self.client.post("/predict", json=transaction_data)
        print(response.text)

    @task
    def update_transaction(self):
        transaction_id = random.randint(1, 1000)  # Assuming transaction IDs are integers
        transaction_data = {
            "merchant": "fraud_Kirlin and Sons",
            "category": "personal_care",
            "amt": 2.86,
            "last": "Elliott",
            "gender": "M",
            "lat": 33.9659,
            "long": -80.9355,
            "city_pop": 333497,
            "job": "Mechanical engineer",
            "merch_lat": 33.986391,
            "merch_long": -81.200714,
            "hour": 3,
            "month": 7
        }

        response = self.client.put(f"/transactions/{transaction_id}", json=transaction_data)
        print(response.text)
