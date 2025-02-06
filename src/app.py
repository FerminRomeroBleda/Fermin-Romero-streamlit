import streamlit as st
import pandas as pd

from pickle import load
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

model = load(open("../models/KNN_neig6_cosine.sav", 'rb'))
df = pd.read_csv("../data/processed/final_table.csv")

if df.empty:
    st.error("Error: El archivo CSV está vacío o no se cargó correctamente.")
else:
    st.write("Archivo CSV cargado con éxito.")

# Rellenar filas NaN
df['tags'] = df['tags'].fillna("").astype(str)

# Vectorización
vmodel = TfidfVectorizer()
vmatrix = vmodel.fit_transform(df['tags'])

st.title('Movie Recommendation System')
st.write('Recommendations based on your favorite movie')

# Input del usuario
movie = st.text_input('Tell me which movie is your favorite')

model = NearestNeighbors(n_neighbors = 6, metric = 'cosine')
model.fit(vmatrix)

similarity = cosine_similarity(vmatrix)


def recommend(movie):

    # Verificar que has introducido el nombre de una película
    if not movie:
        st.error("⚠️ Please enter a movie name")
        return []
    
    # Verificar que la película está en la base de datos
    if movie not in df["title"].values:
        st.error(f"⚠️ The movie '{movie}' was not found in our database. Try another title.")
        return[]
    
    movie_index = df[df["title"] == movie].index[0]

    distances = similarity[movie_index]

    movie_list = sorted(list(enumerate(distances)), reverse = True , key = lambda x: x[1])[1:6]

    # Nombres de las películas recomendadas
    return [df.iloc[i[0]].title for i in movie_list]
    
# Mostrar recomendaciones presionando el botón
if st.button('Recommend'):
    recommendations = recommend(movie)
    
    if recommendations:
        st.write("🎬 **Recommended movies:**")

        for rec in recommendations:
            st.write(f'- {rec}')