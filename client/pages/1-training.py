import streamlit as st
import requests


st.title("Training")

API_BASE_URL = "http://server:8000"
model = st.selectbox(
    "Choose the model",
    ("KNN", "Random Forest Classifier", "Decision Tree Classifier"),
    index=0
)

button_clicked = st.button("Train model")


if button_clicked:
    try:
        with st.spinner('Processing...'):
            response = requests.post(f"{API_BASE_URL}/train/{model}")
            response.raise_for_status()  # Raise an exception for HTTP errors
            result = response.json()  # Parse response as JSON

            # Access elements using dictionary keys
            if result["success"]:
                st.success(result["message"])
                st.markdown(
                    f"[🎯 View {model} in MLFlow]({result['model_link']})",
                    unsafe_allow_html=True
                )
            else:
                st.error(f"Training failed: {result.get('message', 'Unknown error')}")

    except requests.exceptions.RequestException as e:
        st.error(f"An error occurred: {e}")