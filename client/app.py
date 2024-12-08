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

1. **Entraînement de modèles personnalisés :**
   - Choisissez parmi plusieurs algorithmes (KNN, Random Forest, Decision Tree, etc.).
   - Configurez les paramètres d’entraînement directement depuis l’interface.

3. **Suivi et gestion des performances :**
   - Visualisez les métriques d’évaluation (accuracy, RMSE, R²) après chaque entraînement.
   - Enregistrez et gérez vos modèles avec MLflow intégré.

4. **Déploiement et prédictions :**
   - Déployez vos modèles et testez leurs prédictions sur des données réelles.

---

### 🤔 **Comment utiliser l'application ?**

1. **Naviguez dans les différentes sections :**
   - **Exploration des données** : Analysez et préparez vos données.
   - **Entraînement des modèles** : Sélectionnez un algorithme, configurez les paramètres, et lancez l'entraînement.
   - **Suivi des modèles** : Consultez vos résultats et téléchargez vos modèles enregistrés.

2. **Suivi des modèles ML avec MLflow :**
   - Visualisez l’historique des expériences et les artefacts associés.

3. **Tester les prédictions :**
   - Chargez un jeu de données ou saisissez des exemples pour tester vos modèles déployés.

---

### 👨‍💻 **À propos :**

Cette application s'inscrit dans une démarche MLOps, en intégrant des pratiques modernes pour industrialiser et automatiser les workflows de machine learning. Elle est construite avec :
- **Streamlit** : Interface utilisateur conviviale.
- **Docker** : Conteneurisation et scalabilité.
- **MLflow** : Suivi et gestion des modèles.

Explorez, expérimentez et optimisez vos modèles dès aujourd'hui ! 🌍
""")

# Pied de page
st.info("💡 Besoin d'aide ? Contactez l'administrateur ou consultez la documentation.")
