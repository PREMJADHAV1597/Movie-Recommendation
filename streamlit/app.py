# app.py
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------
# Load Data
# ----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Movies Recommendation.csv")
    return df

movies = load_data()

# ----------------------------
# Feature Engineering
# ----------------------------
def combine_features(row):
    return (
        str(row['Movie_Title']) + " " +
        str(row['Movie_Genre']) + " " +
        str(row['Movie_Keywords']) + " " +
        str(row['Movie_Overview']) + " " +
        str(row['Movie_Cast']) + " " +
        str(row['Movie_Director'])
    )

movies['combined_features'] = movies.apply(combine_features, axis=1)

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
    if title not in movies['Movie_Title'].values:
        return []
    idx = movies[movies['Movie_Title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]  # top 10
    movie_indices = [i[0] for i in sim_scores]
    return movies['Movie_Title'].iloc[movie_indices].tolist()

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("🎬 Movie Recommendation System")
st.write("Type a movie name to get recommendations!")

movie_input = st.text_input("Enter a movie title:")

if st.button("Recommend"):
    if movie_input.strip() == "":
        st.warning("⚠️ Please enter a movie title.")
    else:
        recommendations = get_recommendations(movie_input.strip())
        if recommendations:
            st.subheader("Top Recommendations:")
            for i, rec in enumerate(recommendations, start=1):
                st.write(f"{i}. {rec}")
        else:
            st.error("❌ Movie not found in dataset. Try another title.")
