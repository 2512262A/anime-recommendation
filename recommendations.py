import pandas as pd
import re

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix, hstack, vstack

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# NLTK setup

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')


# TEXT HELPERS

stop_words = set(stopwords.words('english'))
lem = WordNetLemmatizer()

def get_base_title(title):
    """
    Normalize anime title to detect franchise/duplicates
    (removes season, movie, ova, etc.)
    """
    title = title.lower()

    title = re.sub(
        r"\b(season\s*\d+|s\d+|part\s*\d+|movie|ova|special|spinoff|spin-off|prequel|sequel)\b",
        "",
        title
    )

    title = re.sub(r"[^a-z0-9 ]", " ", title)
    title = re.sub(r"\s+", " ", title).strip()

    return title

# PREPROCESSING

def preprocess_data(file_path):
    df = pd.read_csv(file_path)

    columns_to_keep = [
        'anime_id', 'Name', 'Score', 'Genres', 'Synopsis',
        'Type', 'Episodes', 'Studios', 'Rating',
        'Popularity', 'Image URL'
    ]
    df = df[columns_to_keep]

    # Remove unwanted types
    df = df[~df['Type'].isin(['Music', 'UNKNOWN', 'Special'])]

    # Clean missing synopsis
    unknown = "No description available for this anime."
    df = df[df['Synopsis'] != unknown]
    df = df[~df['Synopsis'].str.contains("No synopsis has been added", na=False, case=False)]

    # Normalize text
    df['Synopsis'] = df['Synopsis'].str.lower()

    # Clean synopsis (tokenize + stopwords + lemmatize)
    df['Synopsis'] = df['Synopsis'].apply(lambda x: ' '.join(
        [
            lem.lemmatize(word)
            for word in nltk.word_tokenize(x)
            if word.isalpha() and word not in stop_words
        ]
    ))

    # Convert genres/studios into lists
    df['Genres'] = df['Genres'].apply(lambda x: [g.strip() for g in str(x).split(',')])
    df['Studios'] = df['Studios'].apply(lambda x: [s.strip() for s in str(x).split(',')])

    return df

# FEATURE ENCODING

def encode_features(df):
    mlb_genres = MultiLabelBinarizer()
    genres_encoded = csr_matrix(mlb_genres.fit_transform(df['Genres']))

    mlb_studios = MultiLabelBinarizer()
    studios_encoded = csr_matrix(mlb_studios.fit_transform(df['Studios']))

    tfidf = TfidfVectorizer()
    synopsis_encoded = csr_matrix(tfidf.fit_transform(df['Synopsis']))

    # Weighted feature combination
    item_profile = hstack([
        1.5 * genres_encoded,
        1.5 * studios_encoded,
        0.5 * synopsis_encoded
    ])

    return df, item_profile

# USER PROFILE

def create_user_profile(anime_list, df, item_profile):
    idx_list = []

    for anime in anime_list:
        matches = df.index[df['Name'].str.lower() == anime.lower()].tolist()
        if matches:
            idx_list.append(matches[0])
        else:
            raise ValueError(f"Anime '{anime}' not found in dataset.")

    item_profile_csr = item_profile.tocsr()

    user_profile = vstack([item_profile_csr[i] for i in idx_list]).mean(axis=0)

    return csr_matrix(user_profile)



# RECOMMENDATION ENGINE

def recommend(anime_list, df, item_profile):
    df = df.reset_index(drop=True)

    user_profile = create_user_profile(anime_list, df, item_profile)

    sim_matrix = cosine_similarity(user_profile, item_profile)
    sim_scores = sim_matrix[0]
    recs = sim_scores.argsort()[::-1]

    excluded = {a.lower() for a in anime_list}
    input_bases = {get_base_title(a) for a in anime_list}

    filtered = []

    for idx in recs:
        try:
            name = df.loc[idx, 'Name']
            name_lower = name.lower()

            # remove exact matches
            if any(e in name_lower for e in excluded):
                continue

            # remove same franchise (Naruto → Boruto, movies, etc.)
            if get_base_title(name) in input_bases:
                continue

            filtered.append(idx)

            if len(filtered) >= 10:
                break

        except KeyError:
            continue

    return df.iloc[filtered][['Name', 'Genres', 'Synopsis', 'Image URL']]import pandas as pd
import re

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix, hstack, vstack

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# NLTK setup

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')


# TEXT HELPERS

stop_words = set(stopwords.words('english'))
lem = WordNetLemmatizer()

def get_base_title(title):
    """
    Normalize anime title to detect franchise/duplicates
    (removes season, movie, ova, etc.)
    """
    title = title.lower()

    title = re.sub(
        r"\b(season\s*\d+|s\d+|part\s*\d+|movie|ova|special|spinoff|spin-off|prequel|sequel)\b",
        "",
        title
    )

    title = re.sub(r"[^a-z0-9 ]", " ", title)
    title = re.sub(r"\s+", " ", title).strip()

    return title

# PREPROCESSING

def preprocess_data(file_path):
    df = pd.read_csv(file_path)

    columns_to_keep = [
        'anime_id', 'Name', 'Score', 'Genres', 'Synopsis',
        'Type', 'Episodes', 'Studios', 'Rating',
        'Popularity', 'Image URL'
    ]
    df = df[columns_to_keep]

    # Remove unwanted types
    df = df[~df['Type'].isin(['Music', 'UNKNOWN', 'Special'])]

    # Clean missing synopsis
    unknown = "No description available for this anime."
    df = df[df['Synopsis'] != unknown]
    df = df[~df['Synopsis'].str.contains("No synopsis has been added", na=False, case=False)]

    # Normalize text
    df['Synopsis'] = df['Synopsis'].str.lower()

    # Clean synopsis (tokenize + stopwords + lemmatize)
    df['Synopsis'] = df['Synopsis'].apply(lambda x: ' '.join(
        [
            lem.lemmatize(word)
            for word in nltk.word_tokenize(x)
            if word.isalpha() and word not in stop_words
        ]
    ))

    # Convert genres/studios into lists
    df['Genres'] = df['Genres'].apply(lambda x: [g.strip() for g in str(x).split(',')])
    df['Studios'] = df['Studios'].apply(lambda x: [s.strip() for s in str(x).split(',')])

    return df

# FEATURE ENCODING

def encode_features(df):
    mlb_genres = MultiLabelBinarizer()
    genres_encoded = csr_matrix(mlb_genres.fit_transform(df['Genres']))

    mlb_studios = MultiLabelBinarizer()
    studios_encoded = csr_matrix(mlb_studios.fit_transform(df['Studios']))

    tfidf = TfidfVectorizer()
    synopsis_encoded = csr_matrix(tfidf.fit_transform(df['Synopsis']))

    # Weighted feature combination
    item_profile = hstack([
        1.5 * genres_encoded,
        1.5 * studios_encoded,
        0.5 * synopsis_encoded
    ])

    return df, item_profile

# USER PROFILE

def create_user_profile(anime_list, df, item_profile):
    idx_list = []

    for anime in anime_list:
        matches = df.index[df['Name'].str.lower() == anime.lower()].tolist()
        if matches:
            idx_list.append(matches[0])
        else:
            raise ValueError(f"Anime '{anime}' not found in dataset.")

    item_profile_csr = item_profile.tocsr()

    user_profile = vstack([item_profile_csr[i] for i in idx_list]).mean(axis=0)

    return csr_matrix(user_profile)



# RECOMMENDATION ENGINE

def recommend(anime_list, df, item_profile):
    df = df.reset_index(drop=True)

    user_profile = create_user_profile(anime_list, df, item_profile)

    sim_matrix = cosine_similarity(user_profile, item_profile)
    sim_scores = sim_matrix[0]
    recs = sim_scores.argsort()[::-1]

    excluded = {a.lower() for a in anime_list}
    input_bases = {get_base_title(a) for a in anime_list}

    filtered = []

    for idx in recs:
        try:
            name = df.loc[idx, 'Name']
            name_lower = name.lower()

            # remove exact matches
            if any(e in name_lower for e in excluded):
                continue

            # remove same franchise (Naruto → Boruto, movies, etc.)
            if get_base_title(name) in input_bases:
                continue

            filtered.append(idx)

            if len(filtered) >= 10:
                break

        except KeyError:
            continue

    return df.iloc[filtered][['Name', 'Genres', 'Synopsis', 'Image URL']]import streamlit as st
from recommendations import preprocess_data, encode_features, recommend

# Load and preprocess the dataset
st.title("Anime Recommendation System")

@st.cache_data
def load_and_preprocess():
    df = preprocess_data('anime-dataset.csv')
    df, item_profile = encode_features(df)
    return df, item_profile

df, item_profile = load_and_preprocess()

anime_names = sorted(df['Name'].unique())

anime_list = st.multiselect(
    "Choose your favorite anime (max 5):",
    options=anime_names,
    default=[],
    max_selections=5,
    placeholder="Start typing to search..."
)

# Remove duplicates from user input (extra safety)
anime_list = list(dict.fromkeys(anime_list))

if st.button("Get Recommendations"):
    if anime_list:
        try:
            recommendations = recommend(anime_list, df, item_profile)

            # Remove duplicate recommendations (by Name)
            recommendations = recommendations.drop_duplicates(subset=["Name"])

            st.subheader("Top Recommendations:")

            for _, row in recommendations.iterrows():
                st.markdown(f"### {row['Name']}")
                st.image(row['Image URL'], width=200)
                st.markdown(f"**Genres:** {', '.join(row['Genres'])}")
                st.markdown(f"**Synopsis:** {row['Synopsis']}")
                st.write("---")

        except ValueError as e:
            st.error(str(e))
    else:
        st.error("Please select at least one anime.")
