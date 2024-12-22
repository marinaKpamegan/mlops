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
df['target'] = [target_names[t] for t in target] # Ajouter la colonne target au DataFrame

# Affichage du titre
st.title("Visualisation du dataset Iris")

# Afficher les 10 premières lignes du dataset avec les colonnes et le target
st.subheader("Les 10 premières lignes des données Iris avec les colonnes et la cible")
st.dataframe(df.head(10))


# histogramme count plot
fig, ax = plt.subplots()
sns.countplot(x='target', data=df, ax=ax)
ax.set_xticklabels(target_names)
ax.set_xlabel('Species')
ax.set_title('Count plot of Species')

# Scatter plot 3D
fig = px.scatter_3d(
    df, x='sepal length (cm)', y='sepal width (cm)', z='petal length (cm)', color='target'
)
st.plotly_chart(fig)
