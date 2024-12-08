import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn
import pickle
from sklearn.linear_model import LogisticRegression  # for Logistic Regression algorithm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


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
    return rmse, mae, r2

# Fonction pour entraîner et évaluer un modèle
def train_and_evaluate_model(model, model_name, X_train, X_test, y_train, y_test, target_names):
    # Entraînement du modèle
    print("#########################")
    print(X_train.shape, y_train.shape, y_test.shape, X_test.shape)
    model.fit(X_train, y_train)
    
    # Prédictions sur les données de test
    y_pred = model.predict(X_test)
    
    # Calcul des métriques d'évaluation
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy for {model_name}:", accuracy)
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    # Sauvegarde du modèle avec pickle
    # pickle.dump(model, open(f"models/{model_name}.pkl", 'wb'))
    # print(f"Model saved as models/{model_name}.pkl")
    
    # Journalisation avec MLflow
    with mlflow.start_run():
        # Log des métriques supplémentaires
        (rmse, mae, r2) = eval_metrics(y_test, y_pred)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("accuracy", accuracy)
        
        # Log du modèle
        mlflow.sklearn.log_model(model, "model", registered_model_name=model_name)
        print(f"Model logged in MLflow as {model_name}")


def run_model(X, y, target_names, model_type="KNN"):
    X_train, X_test, y_train, y_test = load_data(X, y, test_size=0.4)
    # Exemple d'utilisation avec KNN
    if model_type=="KNN":
        knn = KNeighborsClassifier(n_neighbors=3)
        train_and_evaluate_model(knn, "knn", X_train, X_test, y_train, y_test, target_names)

    # Exemple d'utilisation avec Decision Tree
    if model_type=="Decision Tree Classifier":
        decision_tree = DecisionTreeClassifier(random_state=42)
        train_and_evaluate_model(decision_tree, "decision_tree", X_train, X_test, y_train, y_test, target_names)

    # Exemple d'utilisation avec Random Forest
    if model_type=="Random Forest Classifier":
        random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
        train_and_evaluate_model(random_forest, "random_forest", X_train, X_test, y_train, y_test, target_names)
