import streamlit as st
import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine
from collections import Counter

st.title("Content-based Movie Recommender")

st.write("Upload your MovieLens `movies.csv` and `ratings.csv` files:")

movies_file = st.file_uploader("Upload movies.csv", type=['csv'])
ratings_file = st.file_uploader("Upload ratings.csv", type=['csv'])

if movies_file and ratings_file:
    # Load data
    movies = pd.read_csv(movies_file)
    ratings = pd.read_csv(ratings_file)

    # Genre preprocessing
    movies['genres_list'] = movies['genres'].str.split('|')
    all_genres = [g for sublist in movies['genres_list'] for g in sublist]
    genre_counts = Counter(all_genres)
    N = len(movies)
    idf = {genre: np.log(N / count) for genre, count in genre_counts.items()}

    # Create TF-IDF matrix
    unique_genres = list(idf.keys())
    tfidf_matrix = pd.DataFrame(0.0, index=movies['movieId'], columns=unique_genres)
    for idx, row in movies.iterrows():
        for genre in row['genres_list']:
            tfidf_matrix.loc[row['movieId'], genre] = idf[genre]

    # Compute stats
    movie_stats = ratings.groupby('movieId').agg(
        avg_rating=('rating', 'mean'),
        num_ratings=('rating', 'count')
    ).reset_index()
    global_avg = ratings['rating'].mean()
    m = 10  # Minimum ratings threshold

    movie_stats['weighted_rating'] = (
        (movie_stats['num_ratings'] * movie_stats['avg_rating'] + m * global_avg) /
        (movie_stats['num_ratings'] + m)
    )
    movies = movies.merge(movie_stats, on='movieId', how='left').fillna({'num_ratings': 0, 'avg_rating': 0, 'weighted_rating': global_avg})

    # UI for movie and user selection
    movie_list = movies['title'].sort_values().tolist()
    selected_movie = st.selectbox("Select a movie:", movie_list)
    user_id_input = st.text_input("Enter user ID (optional):")
    top_n = st.slider('Number of recommendations', 5, 20, 10)

    def recommend_similar_movies(query_movie_title, user_id=None, top_n=10):
        query_movie = movies[movies['title'] == query_movie_title]
        if query_movie.empty:
            return "Movie not found."
        query_movie_id = query_movie['movieId'].iloc[0]

        if query_movie_id not in tfidf_matrix.index:
            return "Movie not found."
        query_vector = tfidf_matrix.loc[query_movie_id].values

        similarities = {}
        for movie_id in tfidf_matrix.index:
            if movie_id == query_movie_id:
                continue
            other_vector = tfidf_matrix.loc[movie_id].values
            if np.all(other_vector == 0) or np.all(query_vector == 0):
                sim = 0
            else:
                sim = 1 - cosine(query_vector, other_vector)
            similarities[movie_id] = sim

        candidates = pd.DataFrame(list(similarities.items()), columns=['movieId', 'similarity'])
        candidates = candidates.merge(movies[['movieId', 'title', 'genres_list', 'weighted_rating', 'num_ratings']], on='movieId')
        candidates = candidates[candidates['num_ratings'] >= m]
        candidates['combined_score'] = candidates['similarity'] * candidates['weighted_rating']

        if user_id is not None and user_id != "":
            try:
                user_id_int = int(user_id)
            except Exception:
                st.warning("User ID must be an integer.")
                return []
            user_ratings = ratings[ratings['userId'] == user_id_int]
            seen_movies = user_ratings['movieId'].unique()
            candidates = candidates[~candidates['movieId'].isin(seen_movies)]
            user_seen_similar = []
            for idx, row in candidates.iterrows():
                pred = row['weighted_rating']
                user_seen_similar.append(pred)
            candidates['predicted_rating'] = user_seen_similar
            candidates['combined_score'] *= candidates['predicted_rating'] / global_avg

        top_recs = candidates.sort_values('combined_score', ascending=False).head(top_n)
        query_genres = set(movies[movies['movieId'] == query_movie_id]['genres_list'].iloc[0])
        results = []
        for _, row in top_recs.iterrows():
            shared_genres = query_genres.intersection(set(row['genres_list']))
            explanation = f"Similarity: {row['similarity']:.2f}, Weighted Rating: {row['weighted_rating']:.2f}, Shared Genres: {', '.join(shared_genres)}"
            results.append((row['title'], explanation))
        return results

    if st.button("Show Recommendations"):
        recs = recommend_similar_movies(selected_movie, user_id_input, top_n)
        if isinstance(recs, str):
            st.write(recs)
        elif len(recs) == 0:
            st.write("No recommendations available with current filters.")
        else:
            st.write("Top Recommendations:")
            for title, explanation in recs:
                st.write(f"**{title}** — {explanation}")

else:
    st.warning("Please upload both movies.csv and ratings.csv files.")
