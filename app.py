# app.py — Movie Recommender (Dark Mode, Posters via TMDB)

import pickle
import requests
import streamlit as st
from urllib.parse import quote_plus

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="🎬 Movie Recommender System", page_icon="🎬", layout="wide")

# ---------- DARK THEME CSS ----------
st.markdown("""
<style>
:root{
  --bg:#0B0E14; --panel:#121622; --border:#1d2230;
  --text:#E8EAF0; --muted:#9aa3b2; --chip:#1a2030;
}
[data-testid="stAppViewContainer"]{
  background: radial-gradient(1000px 600px at 120% 120%, #0d1222 0%, #0B0E14 40%),
              radial-gradient(1200px 800px at 10% -10%, #111528 0%, #0B0E14 50%),
              var(--bg);
  color: var(--text);
}
.block-container{ padding-top: 1.4rem; }
h1, h2, h3, h4 { color: var(--text) !important; }
.card{
  border-radius: 14px; padding: 10px 10px 12px;
  background: linear-gradient(180deg, rgba(18,22,34,.96), rgba(18,22,34,.9));
  border:1px solid var(--border);
  box-shadow: 0 6px 24px rgba(0,0,0,.35);
  text-align: center;
}
.card img{ border-radius: 10px; border:1px solid #1b2130; }
.movie-title{ font-weight:700; margin:8px 0 4px; }
.caption{ color:var(--muted); font-size:.85rem; }
</style>
""", unsafe_allow_html=True)

# ---------- TMDB CONFIG ----------
TMDB_KEY = "8946efa099f3906d0252e1115689f7f7"
TMDB_IMG_BASE = "https://image.tmdb.org/t/p/w500"

def fetch_poster(movie_id, title_for_placeholder: str):
    """
    Fetch poster from TMDB; fall back to a placeholder on failure.
    """
    def placeholder(title):
        txt = quote_plus((title or "Movie")[:26])
        return f"https://placehold.co/500x750/0B0E14/E8EAF0?text={txt}"

    try:
        url = f"https://api.themoviedb.org/3/movie/{int(movie_id)}?api_key={TMDB_KEY}&language=en-US"
        resp = requests.get(url, timeout=8)
        resp.raise_for_status()
        data = resp.json()
        poster_path = data.get("poster_path")
        if not poster_path:
            return placeholder(title_for_placeholder)
        return f"{TMDB_IMG_BASE}/{poster_path.lstrip('/')}"
    except Exception:
        return placeholder(title_for_placeholder)

# ---------- LOAD DATA ----------
try:
    movies = pickle.load(open('model/movie_list.pkl','rb'))
    similarity = pickle.load(open('model/similarity.pkl','rb'))
except FileNotFoundError:
    movies = pickle.load(open('movie_list.pkl','rb'))
    similarity = pickle.load(open('similarity.pkl','rb'))

if "movie_id" not in movies.columns and "id" in movies.columns:
    movies = movies.rename(columns={"id": "movie_id"})

# ---------- RECOMMEND FUNCTION ----------
def recommend(movie):
    index = movies[movies['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    recommended_movie_names = []
    recommended_movie_posters = []
    for i in distances[1:6]:
        movie_id = movies.iloc[i[0]].movie_id if 'movie_id' in movies.columns else None
        title = movies.iloc[i[0]].title
        recommended_movie_posters.append(fetch_poster(movie_id, title))
        recommended_movie_names.append(title)
    return recommended_movie_names, recommended_movie_posters

# ---------- UI ----------
st.header('🎬 Movie Recommender System')

movie_list = movies['title'].values
selected_movie = st.selectbox(
    "Type or select a movie from the dropdown",
    movie_list
)

if st.button('Show Recommendation'):
    recommended_movie_names, recommended_movie_posters = recommend(selected_movie)
    cols = st.columns(5)

    for col, name, poster in zip(cols, recommended_movie_names, recommended_movie_posters):
        with col:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.image(poster, use_container_width=True)
            st.markdown(f'<div class="movie-title">{name}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
