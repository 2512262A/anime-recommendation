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
