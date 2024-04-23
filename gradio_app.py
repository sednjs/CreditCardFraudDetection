import gradio as gr
import requests

def predict_fraud(merchant, category, amt, last, gender, lat, long, city_pop, job, merch_lat, merch_long, hour, month):
    url = "http://127.0.0.1:8000/predict"  # URL of API
    data = {
        "merchant": merchant,
        "category": category,
        "amt": amt,
        "last": last,
        "gender": gender,
        "lat": lat,
        "long": long,
        "city_pop": city_pop,
        "job": job,
        "merch_lat": merch_lat,
        "merch_long": merch_long,
        "hour": hour,
        "month": month
    }
    response = requests.post(url, json=data)
    result = response.json()
    return result['is_fraud'], result['probability']

inputs = [
    gr.components.Textbox(label="Merchant"),
    gr.components.Textbox(label="Category"),
    gr.components.Number(label="Amount"),
    gr.components.Textbox(label="Last"),
    gr.components.Dropdown(choices=["M", "F"], label="Gender"),
    gr.components.Number(label="Latitude"),
    gr.components.Number(label="Longitude"),
    gr.components.Number(label="City Population"),
    gr.components.Textbox(label="Job"),
    gr.components.Number(label="Merchant Latitude"),
    gr.components.Number(label="Merchant Longitude"),
    gr.components.Slider(minimum=0, maximum=23, label="Hour"),
    gr.components.Slider(minimum=1, maximum=12, label="Month")
]

outputs = [
    gr.components.Label(label="Is Fraudulent"),
    gr.components.Number(label="Probability")
]

interface = gr.Interface(fn=predict_fraud, inputs=inputs, outputs=outputs, title="Credit Card Fraud Detection", description="Submit transaction details to predict if they are fraudulent.")
interface.launch()

