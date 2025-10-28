# 🎬 Movie Recommender System

A visually striking **Streamlit** web app that recommends movies similar to the one you select.
The model is built using **content-based filtering**, where movie similarity is computed via **text vectorization** and **cosine similarity** between metadata tags. The app uses **TMDB’s API** to fetch real-time posters, wrapped in a sleek **dark UI** interface.

![UI Screenshot](https://drive.google.com/uc?export=view\&id=1mAuq_aQ7OQA58iBGhI1_cN55C_pNjRwK)

---

## 🚀 Features

* 🖤 Dark, minimalist Streamlit design
* 🎥 Top 5 movie recommendations based on content similarity
* 🧠 Pretrained model using **stemming**, **vectorization**, and **cosine similarity**
* 🖼️ Live movie posters via **TMDB API**
* ⚙️ Instant inference using `movie_list.pkl` & `similarity.pkl`

---

## 🧭 How It Works

### 🪄 Step 1 — Data Cleaning & Trimming

Raw TMDB data (e.g. `tmdb_5000_movies.csv`, `tmdb_5000_credits.csv`) contains text-heavy and redundant columns.
We merge and keep only the essentials:

* **title**
* **overview**
* **genres**, **keywords**
* **cast**, **crew (Director)**

All these fields are merged into a single text column — **“tags”** — representing the movie’s content essence.

```python
df['tags'] = df['overview'] + ' ' + df['genres'] + ' ' + df['keywords'] + ' ' + df['cast'] + ' ' + df['director']
```

---

### 🧹 Step 2 — Text Normalization

The text undergoes several NLP preprocessing stages:

1. **Lowercasing**
2. **Removing punctuation & numbers**
3. **Tokenization** — breaking text into individual words
4. **Stopword removal** — filtering out common filler words
5. **Stemming** — reducing words to their root (e.g., *running → run*, *movies → movi*)

This ensures similar concepts map closer in vector space.

---

### 🔢 Step 3 — Vectorization

After preprocessing, we represent every movie as a numerical vector using **CountVectorizer**:

```python
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(df['tags']).toarray()
```

Each column represents a unique word feature; each row represents a movie.

---

### 🧮 Step 4 — Cosine Similarity

To measure how close two movies are based on their text vectors:

[
\text{similarity} = \frac{A \cdot B}{||A|| \times ||B||}
]

This creates a **similarity matrix**, where higher values indicate more alike movies.

The matrix is stored as `similarity.pkl` for fast lookups.

---

### 🧩 Step 5 — Recommendation Function

When a user selects a movie:

1. Its vector row is retrieved.
2. Similarities with all other movies are sorted.
3. Top 5 most similar movies (excluding itself) are returned.

```python
def recommend(movie):
    idx = movies[movies['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[idx])), reverse=True, key=lambda x: x[1])
    return [movies.iloc[i[0]].title for i in distances[1:6]]
```

---

### 🎞️ Step 6 — Poster Fetching (TMDB API)

For each recommendation, the app fetches its poster from TMDB using its movie ID:

```python
url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=YOUR_KEY"
poster_path = response.json()['poster_path']
poster_url = "https://image.tmdb.org/t/p/w500/" + poster_path
```

If TMDB fails or no poster is available, a clean placeholder is displayed instead.

---

### 🖥️ Step 7 — Streamlit Frontend

The `app.py` file handles all UI rendering:

* Header and select box for movie input
* Button trigger for recommendations
* Grid layout of 5 poster cards with titles

The dark gradient CSS gives a cinematic experience to the UI.

---

## 📈 Minimal Flowchart

flowchart TD
    A[Raw TMDB CSVs] --> B[Data Cleaning & Merging]
    B --> C[Tag Creation: overview + genres + keywords + cast + director]
    C --> D[Text Preprocessing: lowercase, stopwords, stemming]
    D --> E[Vectorization (CountVectorizer)]
    E --> F[Cosine Similarity Matrix]
    F --> G[Pickle Files: movie_list.pkl + similarity.pkl]
    G --> H[Streamlit UI (app.py)]
    H --> I[User Selects Movie]
    I --> J[Top-5 Similar Movies + TMDB Posters]
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository

```bash
git clone https://github.com/callmegus4444/movies_recommender_system01.git
cd movies_recommender_system01
```

### 2️⃣ Create and activate a virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
# or
source .venv/bin/activate  # Mac/Linux
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run the app

```bash
streamlit run app.py
```

App will open in your default browser at [http://localhost:8501](http://localhost:8501)

---

## 🧠 Model Files

| File                             | Purpose                                     |
| -------------------------------- | ------------------------------------------- |
| `movie_list.pkl`                 | Contains movie metadata (titles + IDs)      |
| `similarity.pkl`                 | Precomputed cosine similarity matrix        |
| `app.py`                         | Streamlit frontend and logic                |
| `movie-recommender-system.ipynb` | Notebook for model creation & vectorization |

---

## 🧰 Technologies Used

* **Python 3.x**
* **Pandas / NumPy**
* **scikit-learn**
* **Streamlit**
* **TMDB API**

---

## 💡 Future Improvements

* Add genre-based filters and ratings
* Include user collaborative filtering
* Host the app on Streamlit Cloud or Render
* Add caching for TMDB responses

---

## 🧾 License

This project is open-source under the **MIT License**.
