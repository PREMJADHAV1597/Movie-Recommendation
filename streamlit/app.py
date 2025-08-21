import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from thefuzz import process
import requests

# =======================
# Load Dataset
# =======================
movies = pd.read_csv("streamlit/Movies Recommendation.csv")

if "Movie_Title" not in movies.columns:
    st.error("Dataset must have a column named 'Movie_Title'")
    st.stop()

# =======================
# IMDb API Config
# =======================
API_KEY = "YOUR_API_KEY"  # 🔑 Replace with OMDb API key

def get_movie_details(title):
    """
    Fetch movie details (fuzzy search + poster) from OMDb API.
    """
    # First try exact search
    url = f"http://www.omdbapi.com/?t={title}&apikey={API_KEY}"
    data = requests.get(url).json()

    # If exact not found, try fuzzy search using 's=' (search endpoint)
    if data.get("Response") == "False":
        search_url = f"http://www.omdbapi.com/?s={title}&apikey={API_KEY}"
        search_data = requests.get(search_url).json()

        if search_data.get("Response") == "True":
            first_match = search_data["Search"][0]["Title"]
            url = f"http://www.omdbapi.com/?t={first_match}&apikey={API_KEY}"
            data = requests.get(url).json()

    # Return cleaned details
    if data.get("Response") == "True":
        return {
            "title": data.get("Title", "N/A"),
            "rating": data.get("imdbRating", "N/A"),
            "genre": data.get("Genre", "N/A"),
            "plot": data.get("Plot", "N/A"),
            "poster": data.get("Poster", None) if data.get("Poster") != "N/A" else None
        }
    else:
        return {
            "title": title,
            "rating": "N/A",
            "genre": "N/A",
            "plot": "N/A",
            "poster": None
        }

# =======================
# Recommendation Function
# =======================
def recommend(movie, n=5):
    cv = CountVectorizer(stop_words="english")
    count_matrix = cv.fit_transform(movies["Movie_Title"].fillna(""))
    cosine_sim = cosine_similarity(count_matrix)

    idx = movies[movies["Movie_Title"] == movie].index[0]
    scores = list(enumerate(cosine_sim[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    scores = scores[1:n+1]

    movie_indices = [i[0] for i in scores]
    return movies["Movie_Title"].iloc[movie_indices]

# =======================
# Fuzzy Matching Helper
# =======================
def correct_movie_title(user_input):
    choices = movies["Movie_Title"].dropna().unique()
    best_match, score = process.extractOne(user_input, choices)
    return best_match if score >= 60 else None

# =======================
# Streamlit UI
# =======================
st.title("🎬 Movie Recommendation System")

user_input = st.text_input("Enter a movie name:")

if user_input:
    corrected_title = correct_movie_title(user_input)

    if corrected_title:
        st.success(f"🔎 Showing results for: **{corrected_title}**")

        recommendations = recommend(corrected_title, n=5)

        st.subheader("✨ Top Recommendations:")

        for rec in recommendations:
            details = get_movie_details(rec)

            # Display poster if available
            if details["poster"]:
                st.image(details["poster"], width=150)

            st.markdown(f"### {details['title']}")
            st.write(f"⭐ IMDb Rating: {details['rating']}")
            st.write(f"🎭 Genre: {details['genre']}")
            st.write(f"📝 {details['plot']}")
            st.markdown("---")

    else:
        st.error("❌ No close match found for your input. Try again!")
