import streamlit as st
from recommendations import preprocess_data, encode_features, recommend

# Load and preprocess the dataset
st.title("Anime Recommendation System")

@st.cache_data
def load_and_preprocess():
    df = preprocess_data('anime-dataset.csv')
    df, item_profile = encode_features(df)
    return df, item_profile

df, item_profile = load_and_preprocess()

anime_list = st.text_input("Enter a list of your favorite anime (comma-separated):")
anime_list = [anime.strip() for anime in anime_list.split(",") if anime.strip()]

if st.button("Get Recommendations"):
    if anime_list:
        recommendations = recommend(anime_list, df, item_profile)
        st.subheader("Top Recommendations:")
        
        for i, row in recommendations.iterrows():
            st.markdown(f"### {row['Name']}")
            st.image(row['Image URL'], width=200)  # Display anime image
            st.markdown(f"**Genres:** {', '.join(row['Genres'])}")
            st.markdown(f"**Synopsis:** {row['Synopsis']}")
            st.write("---")
    else:
        st.error("Please enter at least one anime.")

