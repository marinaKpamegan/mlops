import streamlit as st
import requests
import base64


st.title("Prediction")

API_BASE_URL = "http://server:8000"
model = st.selectbox(
    "Choose the model",
    ("KNN", "Random Forest Classifier", "Decision Tree Classifier"),
    index=0
)


sepal_length = st.slider("Enter a sepal length", min_value=0.0, max_value=13.0, value=4.8, step=0.1)

sepal_width=st.slider("Enter a sepal width", min_value=0.0, max_value=13.0, value=3.2, step=0.1)


petal_length=st.slider("Enter a petal length", min_value=0.0, max_value=13.0, value=1.9, step=0.1)

petal_width=st.slider("Enter a petal width", min_value=0.0, max_value=13.0, value=0.2, step=0.1)



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
        response = requests.post(f"{API_BASE_URL}/predict/{model}", json=item)
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
                st.image(image_data, caption=f"Image of {prediction}", use_container_width=True)
            else:
                st.error("No image returned from the API.")
        with tab2:
            st.write("Metrics")

        with tab3:
            st.write("Graphs")
            
    except requests.exceptions.RequestException as e:
        st.error(f"An error occurred: {e}")