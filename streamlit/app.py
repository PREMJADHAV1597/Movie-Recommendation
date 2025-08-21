# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------
# Load Data
# ----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("streamlit/Movies Recommendation.csv")  
    return df

movies = load_data()

# ----------------------------
# Feature Engineering
# ----------------------------
def combine_features(row):
    return str(row['title']) + " " + str(row['genres']) + " " + str(row['keywords']) + " " + str(row['cast']) + " " + str(row['director'])

if all(col in movies.columns for col in ['title','genres','keywords','cast','director']):
    movies['combined_features'] = movies.apply(combine_features, axis=1)
else:
    st.error("Your dataset must contain: title, genres, keywords, cast, director")
    st.stop()

# ----------------------------
# Similarity Calculation
# ----------------------------
cv = CountVectorizer(stop_words="english")
count_matrix = cv.fit_transform(movies['combined_features'])
cosine_sim = cosine_similarity(count_matrix)

# ----------------------------
# Recommendation Function
# ----------------------------
def get_recommendations(title, cosine_sim=cosine_sim):
    if title not in movies['title'].values:
        return []
    idx = movies[movies['title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]  # top 10
    movie_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[movie_indices].tolist()

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("🎬 Movie Recommendation System")
st.write("Get recommendations based on your favorite movie")

movie_list = movies['title'].values
selected_movie = st.selectbox("Select a movie:", movie_list)

if st.button("Recommend"):
    recommendations = get_recommendations(selected_movie)
    if recommendations:
        st.subheader("Top Recommendations:")
        for i, rec in enumerate(recommendations, start=1):
            st.write(f"{i}. {rec}")
    else:
        st.warning("Movie not found in dataset.")
