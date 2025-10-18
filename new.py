import streamlit as st
import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

st.title("🎬 Personalized Movie Recommender")
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
    
    # Compute movie stats
    movie_stats = ratings.groupby('movieId').agg(
        avg_rating=('rating', 'mean'),
        num_ratings=('rating', 'count')
    ).reset_index()
    global_avg = ratings['rating'].mean()
    m = 10
    movie_stats['weighted_rating'] = (
        (movie_stats['num_ratings'] * movie_stats['avg_rating'] + m * global_avg) /
        (movie_stats['num_ratings'] + m)
    )
    movies = movies.merge(movie_stats, on='movieId', how='left').fillna({
        'num_ratings': 0, 'avg_rating': 0, 'weighted_rating': global_avg
    })
    
    # Build user-item matrix for collaborative filtering
    st.write("Building user similarity model...")
    user_item_matrix = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)
    
    # Use SVD for dimensionality reduction and user similarity
    svd = TruncatedSVD(n_components=50, random_state=42)
    user_features = svd.fit_transform(user_item_matrix)
    user_similarity = cosine_similarity(user_features)
    user_similarity_df = pd.DataFrame(
        user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index
    )
    
    # UI for movie and user selection
    movie_list = movies['title'].sort_values().tolist()
    selected_movie = st.selectbox("Select a movie:", movie_list)
    user_id_input = st.text_input("Enter user ID (optional for personalization):", value="")
    top_n = st.slider('Number of recommendations', 5, 15, 10)
    
    def predict_user_rating(user_id, movie_id, user_similarity_df, ratings, global_avg):
        """Predict user's rating using collaborative filtering."""
        if user_id not in user_similarity_df.index:
            # Fallback to global weighted rating
            movie_row = movies[movies['movieId'] == movie_id]
            return movie_row['weighted_rating'].iloc[0] if not movie_row.empty else global_avg
        
        # Find similar users
        user_idx = user_similarity_df.index.get_loc(user_id)
        similar_users = user_similarity_df.iloc[user_idx].sort_values(ascending=False).iloc[1:11]
        
        total_weight = 0
        weighted_sum = 0
        
        for sim_user_id, similarity in similar_users.items():
            if similarity < 0.1:  # Similarity threshold
                break
            
            # Get similar user's rating for this movie
            user_ratings = ratings[(ratings['userId'] == sim_user_id) & (ratings['movieId'] == movie_id)]
            if not user_ratings.empty:
                rating = user_ratings['rating'].iloc[0]
                # Adjust for user's average rating bias
                user_avg = ratings[ratings['userId'] == sim_user_id]['rating'].mean()
                adjusted_rating = rating - user_avg + global_avg
                weighted_sum += similarity * adjusted_rating
                total_weight += similarity
        
        if total_weight > 0:
            prediction = global_avg + weighted_sum / total_weight
            return np.clip(prediction, 0.5, 5.0)
        else:
            # Fallback to movie's weighted rating
            movie_row = movies[movies['movieId'] == movie_id]
            return movie_row['weighted_rating'].iloc[0] if not movie_row.empty else global_avg
    
    def recommend_similar_movies(query_movie_title, user_id=None, top_n=10):
        query_movie = movies[movies['title'] == query_movie_title]
        if query_movie.empty:
            return "Movie not found."
        
        query_movie_id = query_movie['movieId'].iloc[0]
        if query_movie_id not in tfidf_matrix.index:
            return "Movie not found."
        
        query_vector = tfidf_matrix.loc[query_movie_id].values
        query_genres = set(query_movie['genres_list'].iloc[0])
        
        # Compute content-based similarities
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
        candidates = candidates.merge(
            movies[['movieId', 'title', 'genres_list', 'weighted_rating', 'num_ratings']], 
            on='movieId'
        )
        candidates = candidates[candidates['num_ratings'] >= m]
        candidates = candidates[candidates['similarity'] >= 0.1]
        
        # Personalization: Exclude seen movies and predict ratings
        if user_id and user_id != "":
            try:
                user_id_int = int(user_id)
                user_ratings = ratings[ratings['userId'] == user_id_int]
                seen_movies = user_ratings['movieId'].unique()
                candidates = candidates[~candidates['movieId'].isin(seen_movies)]
                
                # Compute personalized predicted ratings
                predicted_ratings = []
                for _, row in candidates.iterrows():
                    pred_rating = predict_user_rating(
                        user_id_int, row['movieId'], user_similarity_df, ratings, global_avg
                    )
                    predicted_ratings.append(pred_rating)
                candidates['predicted_rating'] = predicted_ratings
            except ValueError:
                st.warning("Invalid user ID. Using content-based recommendations.")
                candidates['predicted_rating'] = candidates['weighted_rating']
        else:
            candidates['predicted_rating'] = candidates['weighted_rating']
        
        # Compute final scores
        candidates['quality_score'] = candidates['predicted_rating']
        candidates['combined_score'] = candidates['similarity'] * candidates['quality_score']
        
        # Sort and get top recommendations
        top_recs = candidates.sort_values('combined_score', ascending=False).head(top_n)
        
        results = []
        for _, row in top_recs.iterrows():
            shared_genres = query_genres.intersection(set(row['genres_list']))
            explanation = (
                f"Similarity: {row['similarity']:.2f} | "
                f"{'Predicted' if user_id else 'Weighted'} Rating: {row['predicted_rating']:.2f} | "
                f"Shared Genres: {', '.join(list(shared_genres)[:3])}"
            )
            results.append((row['title'], explanation, row['predicted_rating'], row['combined_score']))
        
        return results
    
    if st.button("🎯 Get Personalized Recommendations"):
        with st.spinner("Computing personalized recommendations..."):
            recs = recommend_similar_movies(selected_movie, user_id_input, top_n)
            
            if isinstance(recs, str):
                st.error(recs)
            else:
                st.subheader(f"📺 Top {len(recs)} Recommendations for '{selected_movie}'")
                
                # Show recommendation quality metrics
                if user_id_input and user_id_input != "":
                    try:
                        user_id_int = int(user_id_input)
                        user_rating_count = len(ratings[ratings['userId'] == user_id_int])
                        st.metric("User's Total Ratings", user_rating_count)
                        st.info(f"🔍 Personalized using User {user_id_int}'s rating history and similar users")
                    except:
                        pass
                else:
                    st.info("🌐 Content-based recommendations (no user personalization)")
                
                for i, (title, explanation, pred_rating, score) in enumerate(recs, 1):
                    with st.expander(f"{i}. {title} (Score: {score:.3f})"):
                        st.write(explanation)
                        # Show genre overlap visualization
                        query_genres = set(movies[movies['title'] == selected_movie]['genres_list'].iloc[0])
                        rec_movie = movies[movies['title'] == title]
                        if not rec_movie.empty:
                            rec_genres = set(rec_movie['genres_list'].iloc[0])
                            shared = query_genres.intersection(rec_genres)
                            st.write(f"🎨 **Shared Genres** ({len(shared)}/{len(query_genres)}): {', '.join(list(shared))}")
                
                # Show recommendation statistics
                if len(recs) > 0:
                    scores = [r[3] for r in recs]
                    avg_similarity = np.mean([r[1].split('Similarity: ')[1].split(' |')[0] for r in recs])
                    st.metric("Average Similarity", f"{float(avg_similarity):.2f}")
                    st.metric("Recommendation Range", f"{min(scores):.3f} - {max(scores):.3f}")

else:
    st.warning("📁 Please upload both `movies.csv` and `ratings.csv` files to get started!")
    st.markdown("""
    ### 📋 How to get MovieLens data:
    1. Download from [GroupLens](https://grouplens.org/datasets/movielens/)
    2. Use ML-100K, ML-1M, or ML-25M datasets
    3. Upload `movies.csv` and `ratings.csv` files
    """)
