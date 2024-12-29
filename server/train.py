import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, mean_absolute_error, r2_score, confusion_matrix
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from sklearn.linear_model import LogisticRegression  # for Logistic Regression algorithm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64


# Division des données en train/test
# X_train, X_test, y_train, y_test = None, None, None, None

def load_data(X, y, test_size=0.4):
    X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    return X_train, X_test, y_train, y_test
    

# Fonction pour calculer les métriques
def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    matrix = confusion_matrix(actual, pred)
    return rmse, mae, r2, matrix

# Fonction pour entraîner et évaluer un modèle
def train_and_evaluate_model(model, model_name, X_train, X_test, y_train, y_test, target_names):
    # Entraînement du modèle
    model.fit(X_train, y_train)
    
    # Prédictions sur les données de test
    y_pred = model.predict(X_test)
    
    # Calcul des métriques d'évaluation
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy for {model_name}:", accuracy)
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))


    # Exemple d'entrée pour la signature
    input_example = np.array([X_train[0]])  # Exemple d'entrée (1ère ligne du jeu d'entraînement)
    
    # Génération de la signature
    signature = infer_signature(X_train, y_pred)
    
    # Sauvegarde du modèle avec pickle
    # pickle.dump(model, open(f"models/{model_name}.pkl", 'wb'))
    # print(f"Model saved as models/{model_name}.pkl")
    
    # mlflow.set_experiment("iris_experiment")
    # mlflow.autolog(disable=True)
    # Journalisation avec MLflow
    try:
        with mlflow.start_run():
            # Log des métriques supplémentaires
            (rmse, mae, r2, matrix) = eval_metrics(y_test, y_pred)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2", r2)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("accuracy", accuracy)
            # mlflow.log_metric("confusion_matrix", matrix)


            # Enregistrement de la matrice de confusion comme artefact
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=target_names, yticklabels=target_names)
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title('Confusion Matrix')
            
            # confusion_matrix_path = "confusion_matrix.png"
            
            # Convertir la confusion matrix en base64
            buffer = BytesIO()
            plt.savefig(buffer, format="png")
            buffer.seek(0)
            # cm_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
            cm_base64 = base64.b64encode(buffer.read()).decode("utf-8")
            buffer.close()
            plt.close(fig)
            
            # Log du modèle
            print("###########")
            print(f"Confusion matrix base64 {cm_base64}")

            # Log du modèle avec un exemple d'entrée et une signature
            mlflow.sklearn.log_model(
                sk_model=model, 
                artifact_path="model", 
                registered_model_name=model_name,
                input_example=input_example,
                signature=signature
            )

            print(f"Model logged in MLflow as {model_name}")

            return cm_base64
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
        


def run_model(X, y, target_names, model_type="KNN", test_size=0.4):
    X_train, X_test, y_train, y_test = load_data(X, y, test_size=test_size)
    # Exemple d'utilisation avec KNN
    model = None 
    model_name = ""
    if model_type=="KNN":
        model = KNeighborsClassifier(n_neighbors=3)
        model_name = "knn"

    # Exemple d'utilisation avec Decision Tree
    if model_type=="Decision Tree Classifier":
        model = DecisionTreeClassifier(random_state=42)
        model_name = "decision_tree"
        
    # Exemple d'utilisation avec Random Forest
    if model_type=="Random Forest Classifier":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model_name = "random_forest"

    # training chosen model
    matrix = train_and_evaluate_model(model, model_name, X_train, X_test, y_train, y_test, target_names)
    return model_name, matrix

