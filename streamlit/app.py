# app.py
import streamlit as st
import pandas as pd
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from thefuzz import process  # for fuzzy matching

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
    sim_scores = sim_scores[1:6]  # Top 5 recommendations
    movie_indices = [i[0] for i in sim_scores]
    return movies['Movie_Title'].iloc[movie_indices].tolist()

# ----------------------------
# Fuzzy Matching Function
# ----------------------------
def correct_movie_title(user_input):
    choices = movies['Movie_Title'].dropna().unique()
    best_match, score = process.extractOne(user_input, choices)
    if score > 60:  # threshold for acceptable match
        return best_match
    return None

# ----------------------------
# OMDb API Function
# ----------------------------
API_KEY = "your_api_key_here"  # 🔑 Replace with OMDb API key

def fetch_movie_details(title):
    url = f"http://www.omdbapi.com/?t={title}&apikey={API_KEY}"
    response = requests.get(url).json()
    return response

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("🎬 Smart Movie Recommendation System (with IMDb data)")
st.write("Type a movie name (even with spelling mistakes!) to get recommendations.")

movie_input = st.text_input("Enter a movie title:")

if st.button("Recommend"):
    if movie_input.strip() == "":
        st.warning("⚠️ Please enter a movie title.")
    else:
        # Correct user input with fuzzy matching
        corrected_title = correct_movie_title(movie_input.strip())

        if corrected_title:
            st.success(f"🔍 Did you mean: **{corrected_title}** ?")

            # Fetch details for the selected movie
            details = fetch_movie_details(corrected_title)
            st.subheader("🎥 Selected Movie")
            if details.get("Poster") and details["Poster"] != "N/A":
                st.image(details["Poster"], width=200)
            st.markdown(f"### {details.get('Title', corrected_title)} ({details.get('Year','')})")
            st.write(f"**IMDb Rating:** {details.get('imdbRating','N/A')}")
            st.write(f"**Genre:** {details.get('Genre','N/A')}")
            st.write(f"**Plot:** {details.get('Plot','N/A')}")
            st.markdown("---")

            # Show recommendations
            recommendations = get_recommendations(corrected_title)
            if recommendations:
                st.subheader("✨ Top Recommendations:")
                for rec in recommendations:
                    rec_details = fetch_movie_details(rec)
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        poster = rec_details.get("Poster")
                        if poster and poster != "N/A":
                            st.image(poster, width=100)
                    with col2:
                        st.markdown(f"**{rec_details.get('Title', rec)}** ({rec_details.get('Year','')})")
                        st.write(f"⭐ IMDb Rating: {rec_details.get('imdbRating','N/A')}")
                        st.write(f"🎭 Genre: {rec_details.get('Genre','N/A')}")
                        st.write(f"📝 {rec_details.get('Plot','N/A')}")
                        st.markdown("---")
            else:
                st.warning("No recommendations found.")
        else:
            st.error("❌ No close match found for your input. Try again.")
