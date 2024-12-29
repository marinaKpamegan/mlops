import streamlit as st
import requests
import base64
import mlflow
from mlflow.tracking import MlflowClient

# Configuration
st.title("Prediction")
API_BASE_URL = "http://server:8000"
MLFLOW_TRACKING_URI = "http://mlflow:5000"  # MLFlow server URL
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient()

# Mapping des modèles
model_files = {
    "KNN": "knn",
    "Random Forest Classifier": "random_forest",
    "Decision Tree Classifier": "decision_tree"
}

# Choix du modèle
model = st.selectbox(
    "Choose the model",
    options=list(model_files.keys()),
    index=0
)

# Fonction pour obtenir les versions disponibles pour un modèle donné
def get_model_versions(model_name):
    try:
        model_versions = client.get_registered_model(model_name).latest_versions
        return [version.version for version in model_versions]
    except Exception:
        st.error("No versions available for this model.")
        return None

# Récupération des versions du modèle sélectionné
choosen_model = model_files[model]
model_versions = get_model_versions(choosen_model)

if model_versions:
    # Sélection de la version du modèle
    model_version = st.selectbox("Select a model version", model_versions)

    # Entrée des caractéristiques
    sepal_length = st.slider("Sepal Length", min_value=0.0, max_value=13.0, value=4.8, step=0.1)
    sepal_width = st.slider("Sepal Width", min_value=0.0, max_value=13.0, value=3.2, step=0.1)
    petal_length = st.slider("Petal Length", min_value=0.0, max_value=13.0, value=1.9, step=0.1)
    petal_width = st.slider("Petal Width", min_value=0.0, max_value=13.0, value=0.2, step=0.1)

    # Bouton pour déclencher la prédiction
    if st.button("Predict"):
        # Préparation des données pour la requête
        item = {
            "sepal_length": sepal_length,
            "sepal_width": sepal_width,
            "petal_length": petal_length,
            "petal_width": petal_width
        }

        with st.spinner('Processing...'):
            try:
                # Requête vers l'API
                response = requests.post(f"{API_BASE_URL}/predict/{model}", json=item)
                response.raise_for_status()
                result = response.json()

                # Extraction des résultats
                prediction = result.get("prediction")
                image_base64 = result.get("image")

                # Affichage de la prédiction
                st.subheader("Prediction")
                st.success(f"Prediction: {prediction}")

                # Décodage et affichage de l'image associée
                if image_base64:
                    image_data = base64.b64decode(image_base64)
                    st.image(image_data, caption=f"Image of {prediction}", use_container_width=True)
                else:
                    st.warning("No image returned from the API.")
            except requests.exceptions.RequestException as e:
                st.error(f"An error occurred: {e}")
else:
    st.warning("Please select a valid model and version.")
