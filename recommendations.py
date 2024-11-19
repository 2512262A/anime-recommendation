import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix, hstack, vstack
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import string

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Preprocessing logic
def preprocess_data(file_path):
    df = pd.read_csv(file_path)

    # List of columns to keep
    columns_to_keep = ['anime_id', 'Name', 'Score', 'Genres', 'Synopsis', 'Type',
                       'Episodes', 'Studios', 'Rating', 'Popularity', 'Image URL']
    df = df[columns_to_keep]

    # Filtering and cleaning
    df = df[~df['Type'].isin(['Music', 'UNKNOWN', 'Special'])]
    unknown = "No description available for this anime."
    df = df[df['Synopsis'] != unknown]
    df = df[~df['Synopsis'].str.contains("No synopsis has been added", na=False, case=False)]
    df['Synopsis'] = df['Synopsis'].str.lower()
    
    stop_words = set(stopwords.words('english'))
    lem = WordNetLemmatizer()
    
    # Tokenize, remove stopwords, lemmatize
    df['Synopsis'] = df['Synopsis'].apply(lambda x: ' '.join(
        [lem.lemmatize(word) for word in nltk.word_tokenize(x)
         if word not in stop_words and word not in string.punctuation and word.isalpha()]
    ))
    
    # Preprocess Genres and Studios
    df['Genres'] = df['Genres'].apply(lambda x: [genre.strip() for genre in x.split(',')])
    df['Studios'] = df['Studios'].apply(lambda x: [studio.strip() for studio in x.split(',')])
    
    return df

def encode_features(df):
    # Encode Genres
    mlb = MultiLabelBinarizer()
    genres_encoded = csr_matrix(mlb.fit_transform(df['Genres'].tolist()))
    
    # Encode Studios
    studios_encoded = csr_matrix(mlb.fit_transform(df['Studios'].tolist()))
    
    # Encode Synopsis
    tfidf = TfidfVectorizer()
    synopsis_encoded = csr_matrix(tfidf.fit_transform(df['Synopsis']))
    
    # Weight and combine features
    item_profile = hstack([1.5 * genres_encoded, 1.5 * studios_encoded, 0.5 * synopsis_encoded])
    
    return df, item_profile

def create_user_profile(anime_list, df, item_profile):
    idx_list = []
    
    for anime in anime_list:
        matches = df.index[df['Name'].str.lower() == anime.lower()].tolist()
        if matches:
            idx_list.append(matches[0])
        else:
            raise ValueError(f"Anime '{anime}' not found in the dataset. Please check the name.")
    
    if not idx_list:
        raise ValueError("No valid anime found in the list. Please try again.")

    item_profile_csr = item_profile.tocsr()
    user_profile = vstack([item_profile_csr[idx] for idx in idx_list]).mean(axis=0)
    
    return csr_matrix(user_profile)

def recommend(anime_list, df, item_profile):
    df = df.reset_index(drop=True)
    
    user_profile = create_user_profile(anime_list, df, item_profile)
    sim_matrix = cosine_similarity(user_profile, item_profile)
    sim_scores = sim_matrix[0]
    recs = sim_scores.argsort()[::-1]

    excluded_keywords = {anime.lower() for anime in anime_list}
    filtered_recs = []

    for rec in recs:
        try:
            rec_anime = df.loc[rec, 'Name'].lower()
            if any(keyword in rec_anime for keyword in excluded_keywords):
                continue
            filtered_recs.append(rec)
            if len(filtered_recs) >= 10:
                break
        except KeyError:
            continue

    # Return a DataFrame with the top recommendations
    return df.iloc[filtered_recs][['Name', 'Genres', 'Synopsis', 'Image URL']]
