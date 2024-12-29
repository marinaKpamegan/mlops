import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.datasets import load_iris
import pandas as pd
import seaborn as sns

# Charger les données iris
iris = load_iris()
data = iris.data
target = iris.target
columns = iris.feature_names
target_names = iris.target_names

# Convertir en DataFrame pour un meilleur affichage
df = pd.DataFrame(data, columns=columns)
df['target'] = [target_names[t] for t in target]  # Ajouter la colonne target au DataFrame

# Affichage du titre
st.title("Visualisation du dataset Iris")

# Onglets pour les visualisations
tab1, tab2 = st.tabs(["Scatterplot 3D", "Countplot"])

with tab1:
    # Scatter plot 3D
    st.subheader("Visualisation en 3D")
    fig = px.scatter_3d(
        df, x='sepal length (cm)', y='sepal width (cm)', z='petal length (cm)', color='target',
        title="Scatterplot 3D des Iris"
    )
    st.plotly_chart(fig)

with tab2:
    # Count plot
    st.subheader("Répartition des espèces dans le dataset")
    fig, ax = plt.subplots()
    sns.countplot(x='target', data=df, ax=ax, palette="viridis")
    ax.set_xticklabels(target_names)
    ax.set_xlabel('Espèce')
    ax.set_ylabel('Nombre d\'échantillons')
    ax.set_title('Count plot des espèces')
    st.pyplot(fig)
