import streamlit as st
import requests


st.title("All predictions")

API_BASE_URL = "http://server:8000"


# Get all predictions
response = requests.get(f"{API_BASE_URL}/get-predictions")
response.raise_for_status()


# Parse the response JSON to get predictions
predictions = response.json().get("predictions", [])


# Prepare data for the table
if predictions:
    # Prepare a list of dictionaries for each prediction to display in the table
    predictions_table = [{
        "Model": prediction["model"],
        "Model URI": f"[{prediction['model_uri']}]({prediction['model_uri']})",  # Create markdown-style link as string
        "Input": str(prediction["input"]),  # Convert input data to string for display
        "Prediction": prediction["prediction"],
        "Timestamp": prediction["timestamp"]
    } for prediction in predictions]
    
    # Display the predictions as a table in Streamlit
    st.table(predictions_table)