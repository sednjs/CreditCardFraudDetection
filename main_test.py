import warnings
import pytest
from fastapi.testclient import TestClient
from main import app

warnings.filterwarnings("ignore", category=DeprecationWarning)

client = TestClient(app)

def run_test(test_func):
    try:
        test_func()
        print(f"{test_func.__name__} - PASSED")
    except AssertionError as e:
        print(f"{test_func.__name__} - FAILED: {str(e)}")

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
    

# Test for invalid input types
def test_invalid_input_types():
    response = client.post("/predict", json={
        "merchant": "fraud_Kirlin and Sons",
        "category": "personal_care",
        "amt": "invalid_amount",  # this should be a float
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
    assert response.status_code == 422  # expecting a failure due to type error

def test_missing_fields():
    response = client.post("/predict", json={
        # 'merchant' field is missing
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
    assert response.status_code == 422  # FastAPI typically returns 422 for validation errors

def test_boundary_values():
    # Test lower and upper boundaries for 'hour' and 'month'
    test_cases = [
        {"hour": -1, "month": 0},  # Both values are below minimum
        {"hour": 0, "month": 1},   # Lower boundary correct values
        {"hour": 23, "month": 12}, # Upper boundary correct values
        {"hour": 24, "month": 13}, # Both values are above maximum
    ]

    for case in test_cases:
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
            **case  # Expand the test case dictionary here
        })
        print(response.status_code, response.json())  # Optional: Print to debug test outputs
        if case["hour"] in [-1, 24] or case["month"] in [0, 13]:
            assert response.status_code == 422, f"Failed at case: {case}"
        else:
            assert response.status_code == 200, f"Failed at case: {case}"

def test_invalid_data_types():
    # Sending a string instead of a float for "amt"
    response = client.post("/predict", json={
        "merchant": "fraud_Kirlin and Sons",
        "category": "personal_care",
        "amt": "wrong_type",  # Invalid type: string instead of float
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
    assert response.status_code == 422  # Expecting a failure due to type mismatch
    assert "float_parsing" in response.text, "Expected float parsing error not found in response"
    assert "loc" in response.text and "amt" in response.text, "Location of the error should specify the 'amt' field"
    assert "unable to parse string" in response.text, "Error message not as expected"

def test_boundary_values():
    # Define a set of test transactions with boundary values
    test_cases = [
        # Maximum values
        {
            "merchant": "fraud_Kirlin and Sons",
            "category": "personal_care",
            "amt": 28948.9,  # Maximum amt
            "last": "Elliott",
            "gender": "M",
            "lat": 66.6933,  # Maximum lat
            "long": -67.9503,  # Maximum long
            "city_pop": 100000,
            "job": "Mechanical engineer",
            "merch_lat": 66.6933,
            "merch_long": -67.9503,
            "hour": 23,
            "month": 12
        },
        # Minimum values
        {
            "merchant": "fraud_Kirlin and Sons",
            "category": "personal_care",
            "amt": 1.0,  # Minimum amt
            "last": "Elliott",
            "gender": "M",
            "lat": 20.0271,  # Minimum lat
            "long": -165.6723,  # Minimum long
            "city_pop": 500,
            "job": "Mechanical engineer",
            "merch_lat": 20.0271,
            "merch_long": -165.6723,
            "hour": 0,
            "month": 1
        }
    ]

    for case in test_cases:
        response = client.post("/predict", json=case)
        assert response.status_code == 200, f"Failed at case: {case}"
        response_data = response.json()
        assert 'is_fraud' in response_data and 'probability' in response_data, "Response data is incomplete."

if __name__ == "__main__":
    # List all tests to run
    tests = [
        test_read_main,
        test_predict_fraud,
        test_invalid_input_types,
        test_missing_fields,
        test_boundary_values,
        test_invalid_data_types
    ]
    # Run all tests
    for test in tests:
        run_test(test)
