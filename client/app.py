import streamlit as st

# Configuration de la page
st.set_page_config(page_title="Machine Learning MLOps Platform", page_icon="🚀", layout="wide")

# Titre principal
st.title("Bienvenue sur votre plateforme de Machine Learning MLOps 🚀")

# Sous-titre
st.subheader("Simplifiez, automatisez et suivez vos modèles de Machine Learning avec cette application intuitive.")

# Contenu principal
st.markdown("""
### 🌟 **Fonctionnalités principales :**

1. **Entraînement de modèles personnalisés sur les données Iris :**
   - Cette plateforme utilise les **données Iris de scikit-learn** pour entraîner les modèles.
   - Choisissez parmi plusieurs algorithmes (KNN, Random Forest, Decision Tree).
   - Configurez la taille du jeu de test via le paramètre `test_size` (par défaut 0.4).
   - L'application se charge de la préparation des données, vous n'avez pas à vous en soucier !

2. **Visualisation des données :**
   - Visualisez les données Iris avec des graphiques interactifs (scatter plot 3D, count plot).

3. **Suivi et gestion des performances :**
   - Visualisez les métriques d’évaluation (accuracy, RMSE, R²) après chaque entraînement.
   - Enregistrez et gérez vos modèles avec **MLflow** intégré pour un suivi complet.

4. **Prédictions en temps réel :**
   - Testez les prédictions en temps réel avec un modèle déjà entraîné.
   - Visualisez les résultats et les images associées aux prédictions.
   - Une base de données MongoDB est utilisée pour stocker les prédictions et les modèles utilisés.

--- 


""")

