from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"msg": "Hello! This is the credit card fraud prediction ML service!"}

@app.get("/")
def root():
    return {"msg": "Hello! This is the credit card fraud prediction ML service!"}


def test_predict_fraud():
    response = client.post("/predict", json={
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
    })
    assert response.status_code == 200
    assert "is_fraud" in response.json()
    assert "probability" in response.json()

