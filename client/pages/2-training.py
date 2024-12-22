import streamlit as st
import requests
import mlflow
import base64
from mlflow.tracking import MlflowClient


st.title("Training")

API_BASE_URL = "http://server:8000"
MLFLOW_TRACKING_URI = "http://mlflow:5000"  # MLFlow server URL
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient()
model_files = {"KNN":"knn", "Random Forest Classifier":"random_forest", "Decision Tree Classifier":"decision_tree"}


model = st.selectbox(
    "Choose the model",
    ("KNN", "Random Forest Classifier", "Decision Tree Classifier"),
    index=0
)

test_size = st.slider("Select test size", 0.0, 1.0, 0.4, step=0.1)


button_clicked = st.button("Train model")


if button_clicked:
    try:
        with st.spinner('Processing...'):
            response = requests.post(f"{API_BASE_URL}/train/{model}/{test_size}")
            response.raise_for_status()  # Raise an exception for HTTP errors
            result = response.json()  # Parse response as JSON

            # st.write(result)
            # Access elements using dictionary keys
            if result["success"]:
                st.success(result["message"])
                st.markdown(
                    f"[🎯 View {model} in MLFlow]({result['model_link']})",
                    unsafe_allow_html=True
                )
                tab1, tab2 = st.tabs(["Metrics", "Graph"])
                with tab1:
                    try:
                        # Récupérer la dernière version du modèle
                        latest_versions = client.get_latest_versions(model_files[model], stages=["None", "Production", "Staging"])
                        if not latest_versions:
                            st.error(f"Aucune version trouvée pour le modèle : {model_files[model]}")
                            st.stop()

                        # Trier pour obtenir la dernière version
                        latest_version = max(latest_versions, key=lambda v: int(v.version))
                        run_id = latest_version.run_id
                        version_number = latest_version.version

                        # Récupération des métriques associées à ce Run ID
                        run = client.get_run(run_id)
                        metrics = run.data.metrics

                        # Affichage des métriques
                        st.subheader(f"Métriques pour {model} (Version {version_number})")

                        # Convertir les métriques en tableau (liste de dictionnaires ou DataFrame)
                        metrics_table = [{"Metric": metric_name, "Value": metric_value} for metric_name, metric_value in metrics.items()]

                        # Afficher les métriques sous forme de tableau
                        st.table(metrics_table)


                        # Affichage des informations sur le Run
                        # st.write(f"Run ID associé : {run_id}")
                        # st.write(f"Statut du modèle : {latest_version.current_stage}")

                    except Exception as e:
                        st.error(f"An error occurred: {e}")


                with tab2:
                    if result["matrix_base64"]:
                        image_data = base64.b64decode(result["matrix_base64"])
                        st.image(image_data, caption=f"Image of confusion matrix", use_container_width=True)

            else:
                st.error(f"Training failed: {result.get('message', 'Unknown error')}")

    except requests.exceptions.RequestException as e:
        st.error(f"An error occurred: {e}")





