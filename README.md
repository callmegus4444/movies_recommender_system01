# 🎮 Movie Recommender System

A clean, dark-mode **Streamlit** app that recommends movies similar to the one you select.
It uses classic **content-based filtering**: text preprocessing (tokenization, stop-words removal, stemming), **vectorization** (CountVectorizer or TF-IDF), and **cosine similarity** between movie “tag” vectors. Posters are fetched from **TMDB**.

![App UI](https://drive.google.com/uc?export=view\&id=1mAuq_aQ7OQA58iBGhI1_cN55C_pNjRwK)

---

## ✨ Features

* Minimal, elegant **dark UI** in Streamlit
* **Type & choose** a movie → get **top-5 similar** titles
* **Posters** pulled from TMDB (fallback to a placeholder if missing)
* Precomputed `movie_list.pkl` (movie metadata) and `similarity.pkl` (cosine matrix) for instant startup
* Simple, hackable codebase

---

## 📁 Project Structure

```
movies_recommender_system01/
├─ app.py                 # Streamlit UI + recommendations + TMDB posters
├─ movie_list.pkl         # DataFrame: at least ['title', 'movie_id', ...]
├─ similarity.pkl         # 2D array: cosine similarity between movies
├─ README.md
└─ (optional) tmdb_5000_movies.csv / tmdb_5000_credits.csv  # for re-building the model
```

> **Note:** `similarity.pkl` can be very large. Don’t commit >100 MB files to GitHub; see the troubleshooting section below.

---

## 🚀 Quickstart

### 1) Create & activate a virtual environment

```bash
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\activate
# Mac/Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install streamlit pandas numpy scikit-learn requests
```

### 3) Put model files in place

* Ensure `movie_list.pkl` and `similarity.pkl` are in the repo root (or in `model/`—the app looks in both).

### 4) (Optional) TMDB API key

The current app **hard-codes** the TMDB key (replace it if you want). Posters will work out-of-the-box.

> Prefer storing the key in `.streamlit/secrets.toml` for production.

### 5) Run the app

```bash
streamlit run app.py
```

Open the browser at `http://localhost:8501`. Select a movie → hit **Show Recommendation**.

---

## 🧠 How It Works (Model)

### 1) Data sources

* **TMDB 5000 datasets** (`tmdb_5000_movies.csv`, `tmdb_5000_credits.csv`)
* We create a **compact movie table** (title, id, genres, keywords, cast, crew, overview).

### 2) Build “tags” per movie

Concatenate salient text fields:

* `overview` (short description),
* `genres`, `keywords`,
* top **N** cast members,
* director name (from crew)
  → join into one text blob: **tags**.

### 3) Text cleaning & normalization

* Lowercasing
* Remove punctuation & non-letters
* **Tokenization** (split into words)
* **Stop-word removal** (e.g., English stop words)
* **Stemming** (e.g., Porter stemmer) to reduce words to roots

  * “running”, “runs” → “run”
  * “movies”, “movie” → “movi”

This step reduces sparsity and helps group similar words.

### 4) Vectorization

Convert `tags` to numeric vectors with either:

* **CountVectorizer** (bag-of-words; `max_features=5000`, `stop_words='english'`)
* or **TF-IDF** (weighs rare terms higher, often improves quality)

Result: an **M × F** matrix (M movies, F features). Each movie → one vector.

### 5) Similarity matrix

Compute **cosine similarity** between all movie vectors:

[
\text{cosine_sim}(\mathbf{a}, \mathbf{b})
= \frac{\mathbf{a} \cdot \mathbf{b}}{|\mathbf{a}|;|\mathbf{b}|}
]

* Values range **0 → 1** (higher = more similar).
* Store the **full M×M** cosine matrix as `similarity.pkl`.

### 6) Runtime query

* User picks a **movie**
* Find its row index `i`
* Sort row `similarity[i]` in descending order
* Pick the **top-5** different indices → show titles & posters

---

## 🗮️ Rebuilding `movie_list.pkl` & `similarity.pkl` (from CSVs)

Below is a **reference notebook flow** (keep in your `.ipynb`). You can adapt it to your columns.

```python
import pandas as pd, numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast, re, pickle
from nltk.stem.porter import PorterStemmer

# 1) load
movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')
df = movies.merge(credits, on='title')

# 2) helpers
def parse_names(x):
    try:
        return [d['name'] for d in ast.literal_eval(x)]
    except Exception:
        return []

def top_cast(x, k=3):
    try:
        return [d['name'] for d in ast.literal_eval(x)][:k]
    except Exception:
        return []

def get_director(x):
    try:
        for d in ast.literal_eval(x):
            if d.get('job') == 'Director':
                return d.get('name','')
    except Exception:
        pass
    return ''

# 3) build tags
df['genres']   = df['genres'].apply(parse_names)
df['keywords'] = df['keywords'].apply(parse_names)
df['cast']     = df['cast'].apply(top_cast)
df['director'] = df['crew'].apply(get_director)

def clean(s):
    s = re.sub(r'[^a-zA-Z\s]', ' ', str(s).lower())
    return re.sub(r'\s+', ' ', s).strip()

df['tags'] = (
    df['overview'].fillna('') + ' ' +
    df['genres'].apply(lambda x: ' '.join(x)) + ' ' +
    df['keywords'].apply(lambda x: ' '.join(x)) + ' ' +
    df['cast'].apply(lambda x: ' '.join(x)) + ' ' +
    df['director'].fillna('')
).apply(clean)

# 4) stemming
ps = PorterStemmer()
df['tags'] = df['tags'].apply(lambda s: ' '.join(ps.stem(w) for w in s.split()))

# 5) vectorize + cosine
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(df['tags']).toarray()
sim = cosine_similarity(vectors)

# 6) keep columns needed by the app
movie_list = df[['title', 'id']].rename(columns={'id': 'movie_id'})

# 7) save pickles
pickle.dump(movie_list, open('movie_list.pkl','wb'))
pickle.dump(sim, open('similarity.pkl','wb'))
```

> If you use **TF-IDF** instead of CountVectorizer, replace the import and the vectorizer call accordingly.

---

## 🖼️ Posters (TMDB)

* The app calls TMDB’s movie endpoint and constructs a poster URL from `poster_path`.
* If TMDB fails or a path is missing, the app shows a **clean text placeholder** instead.
* In this version, the key is **hard-coded** in `app.py` (change it or move to secrets in production).

---

## 🤩 Streamlit UI (what’s in `app.py`)

* **Selectbox** to pick a movie title
* **Button** to trigger recommendation
* **5-column** responsive grid of poster cards (title + image)
* The app looks for `movie_list.pkl`/`similarity.pkl` in root or `model/` and shows results immediately.

---

## 🗂️ Flowchart (end-to-end)

```mermaid
flowchart TD
    A[TMDB 5000 CSVs] --> B[Merge & Select Columns]
    B --> C[Build Tags: overview + genres + keywords + cast + director]
    C --> D[Clean/Normalize: lowercase, remove punct, stopwords, stemming]
    D --> E[Vectorize (CountVectorizer or TF-IDF)]
    E --> F[Cosine Similarity Matrix]
    F --> G[Save movie_list.pkl & similarity.pkl]
    G --> H[Streamlit App (app.py)]
    H --> I[User selects movie]
    I --> J[Top-5 similar (cosine row sort)]
    J --> K[Fetch Poster (TMDB)]
    K --> L[Display Cards in Dark UI]
```

---

## 🤮 Example Recommendation Logic

```python
def recommend(title, k=5):
    idx = movie_list[movie_list['title'] == title].index[0]
    # enumerate & sort by similarity descending (skip itself)
    distances = sorted(list(enumerate(similarity[idx])),
                       reverse=True, key=lambda x: x[1])
    picks = distances[1:k+1]
    return [movie_list.iloc[i].title for i, _ in picks]
```

---

## 🛠️ Troubleshooting

* **GitHub rejects `similarity.pkl` (>100 MB)**

  * Don’t commit it. Add to `.gitignore`: `*.pkl`
  * Host the file elsewhere (Drive/S3/HF) or use **Git LFS**
* **Streamlit image deprecation**

  * Newer Streamlit prefers `width='stretch'` over `use_container_width=True`
* **TMDB key security**

  * Move the key to `.streamlit/secrets.toml` in production:

    ```
    TMDB_API_KEY="YOUR_KEY"
    ```
  * Then read via `st.secrets["TMDB_API_KEY"]`

---

## 📜 License

MIT — use it, modify it, have fun.
