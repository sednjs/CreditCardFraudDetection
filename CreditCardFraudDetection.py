# Install required packages
# import sys
# import subprocess
# subprocess.check_call([sys.executable, "-m", "pip", "install", "gradio"])



import joblib
import pandas as pd
import gradio as gr

# Load the trained model
best_pipeline = joblib.load("best_pipeline.joblib")

def predict_fraud(merchant, category, amt, last, gender, lat, long, city_pop, job, merch_lat, merch_long, hour, month):
    data = {
        'merchant': [merchant],
        'category': [category],
        'amt': [amt],
        'last': [last],
        'gender': [gender],
        'lat': [lat],
        'long': [long],
        'city_pop': [city_pop],
        'job': [job],
        'merch_lat': [merch_lat],
        'merch_long': [merch_long],
        'hour': [hour],
        'month': [month]
    }
    input_data = pd.DataFrame(data)
    
    # Predict
    prediction = best_pipeline.predict(input_data)
    probability = best_pipeline.predict_proba(input_data)[:,1]

    return "Fraudulent" if bool(prediction) else "Not Fraudulent", float(probability)

# Create input components
inputs = [
    gr.Textbox(label="Merchant"),
    gr.Textbox(label="Category"),
    gr.Number(label="Amount"),
    gr.Textbox(label="Last"),
    gr.Textbox(label="Gender"),
    gr.Number(label="Latitude"),
    gr.Number(label="Longitude"),
    gr.Number(label="City Population"),
    gr.Textbox(label="Job"),
    gr.Number(label="Merchant Latitude"),
    gr.Number(label="Merchant Longitude"),
    gr.Number(label="Hour"),
    gr.Number(label="Month"),
]

# Create interface
interface = gr.Interface(
    fn=predict_fraud,
    inputs=inputs,
    outputs=["text", "text"],
    title="Credit Card Fraud Detection",
    description="Enter transaction details to predict if it's fraudulent.",
    examples=[
        ["OnlineRetail", "Retail", 100.0, "Purchase", "M", 37.235, -115.806, 3495, "Artist", 38.939, -78.814, 0, 1],
        ["ElectronicsStore", "Electronics", 200.0, "Purchase", "F", 40.712, -74.006, 8175, "Engineer", 40.712, -74.006, 12, 3],
    ],
)

# Launch interface
interface.launch()
