import streamlit as st
import pickle
import pandas as pd
import numpy as np
from fuzzywuzzy import process

st.title("Movie Recommendation System")
movie_df = pickle.load(open("movie.pkl", "rb"))
recommender_model = pickle.load(open("recommender_model.pkl", "rb"))
mat_movies = pickle.load(open("mat_movies.pkl", "rb"))

# Contenu de l'option
list_movie = np.array(movie_df["title"])

def recommender(movie_name, data, n):
    # Code de recherche de l'index du film
    idx = process.extractOne(movie_name, movie_df['title'])[2]
   
    st.write("Recherche de recommandations...")

    # Transformer en un tableau 2D avec reshape(-1, 1)
    distances, indices = recommender_model.kneighbors(data[idx].reshape(1, -1), n_neighbors=n)
    recommended_movies = movie_df['title'].where(np.isin(np.arange(len(movie_df['title'])), indices))
    recommended_movies = recommended_movies.dropna()  # Retirer les valeurs nulles

    # Afficher les recommandations dans Streamlit
    for movie in recommended_movies.values:
        st.write(movie)

# Option de sélection
option = st.selectbox("Select Movie ", list_movie)

if st.button('Recommend Me'):
    st.write('Movies Recommended for you are:')
    recommender(option, mat_movies, 10)
