import streamlit as st
import requests
import base64

API_BASE_URL = "http://server:8000"

st.title("Iris Dataset Prediction")

model = st.selectbox(
    "Choose the model",
    ("KNN", "Random Forest", "Decision Tree"),
    index=0
)


sepal_length=st.number_input(
        "Enter a sepal length",
        4.8,
        key="sepal_length",
    )

sepal_width=st.number_input(
        "Enter a sepal width",
        3.2,
        key="sepal_width",
    )


petal_length=st.number_input(
        "Enter a petal length",
        1.9,
        key="petal_length",
    )

petal_width=st.number_input(
        "Enter a petal width",
        0.2,
        key="petal_width",
    )

button_clicked = st.button("Predict")
tab1, tab2, tab3 = st.tabs(["Prediction", "Metrics", "Graph"])

if button_clicked:
    item = {
        "sepal_length": sepal_length,
        "sepal_width": sepal_width,
        "petal_length": petal_length,
        "petal_width": petal_width
    }
    
    try:
        response = requests.post(f"{API_BASE_URL}/predict/", json=item)
        response.raise_for_status()
        result = response.json()

        # Extract prediction and Base64 image
        prediction = result.get("prediction")
        image_base64 = result.get("image")

        with tab1:
            st.success(f"Prediction: {prediction}")
            # Decode and display the image
            if image_base64:
                image_data = base64.b64decode(image_base64)
                st.image(image_data, caption=f"Image of {prediction}", use_column_width=True)
            else:
                st.error("No image returned from the API.")
        with tab2:
            st.write("Metrics")

        with tab3:
            st.write("Graphs")
            
    except requests.exceptions.RequestException as e:
        st.error(f"An error occurred: {e}")