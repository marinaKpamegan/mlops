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

2. **Suivi et gestion des performances :**
   - Visualisez les métriques d’évaluation (accuracy, RMSE, R²) après chaque entraînement.
   - Enregistrez et gérez vos modèles avec **MLflow** intégré pour un suivi complet.

--- 

### 🤔 **Comment utiliser l'application ?**

1. **Naviguez dans les différentes sections :**
   - **Entraînement des modèles** : Sélectionnez un algorithme parmi ceux proposés, ajustez la taille du jeu de test et lancez l'entraînement.
   - **Suivi des modèles** : Consultez vos résultats (précision, RMSE, etc.) après chaque entraînement.
   - **Tester les prédictions** : Utilisez un modèle déjà entraîné pour tester des prédictions sur un échantillon de données.

2. **Tester les prédictions :**
   - Chargez un modèle et testez des prédictions directement depuis l’interface.

--- 

### 👨‍💻 **À propos :**

Cette application s'inscrit dans une démarche **MLOps**, en intégrant des pratiques modernes pour industrialiser et automatiser les workflows de machine learning. Elle est construite avec :
- **Streamlit** : Interface utilisateur conviviale.
- **Docker** : Conteneurisation et scalabilité.
- **MLflow** : Suivi, gestion des modèles et des expériences.

---

""")

