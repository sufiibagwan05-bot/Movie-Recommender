import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Page Configuration (Wider layout looks better for grids)
st.set_page_config(page_title="Movie Recommender", page_icon="🍿", layout="wide")

st.title("🍿 AI Movie Recommendation Engine")
st.write("Select a movie you love, and the AI will recommend 5 similar movies!")

# 2. Load and Process Data
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/rashida048/Some-NLP-Projects/master/movie_dataset.csv"
    df = pd.read_csv(url)
    
    features = ['keywords', 'cast', 'genres', 'director']
    for feature in features:
        df[feature] = df[feature].fillna('')
        
    def combine_features(row):
        return row['keywords'] + " " + row['cast'] + " " + row['genres'] + " " + row['director']
        
    df['combined_features'] = df.apply(combine_features, axis=1)
    
    cv = CountVectorizer()
    count_matrix = cv.fit_transform(df['combined_features'])
    cosine_sim = cosine_similarity(count_matrix)
    
    return df, cosine_sim

df, cosine_sim = load_data()

# 3. Interactive Web Interface 
# Streamlit automatically makes this a searchable dropdown!
movie_list = df['title'].dropna().values
selected_movie = st.selectbox("Search or select a movie from the dropdown:", movie_list)

if st.button("Get Recommendations"):
    movie_index = df[df.title == selected_movie]["index"].values[0]
    
    similar_movies = list(enumerate(cosine_sim[movie_index]))
    sorted_similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)[1:6]
    
    st.markdown("---")
    st.subheader(f"Because you liked **{selected_movie}**, you should watch:")
    
    # --- THE ATTRACTIVE UPGRADE: Grid Layout ---
    # Create 5 side-by-side columns
    cols = st.columns(5)
    
    for i, movie in enumerate(sorted_similar_movies):
        # Extract data for the recommended movie
        title = df[df.index == movie[0]]["title"].values[0]
        genres = df[df.index == movie[0]]["genres"].values[0]
        director = df[df.index == movie[0]]["director"].values[0]
        
        # Put each movie in its own column box
        with cols[i]:
            # st.info creates a nice colored box around the title
            st.info(f"**{title}**")
            
            # st.expander creates a clickable dropdown for extra info
            with st.expander("View Details"):
                st.caption(f"**Director:** {director}")
                st.caption(f"**Genres:** {genres}")