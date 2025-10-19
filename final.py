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
    movies['genres_list'] = movies['genres'].str.split('|').apply(lambda x: x if isinstance(x, list) and x else ['(no genres listed)'])
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
    diversity_weight = st.slider('Diversity Weight (0 = no diversity, 1 = max diversity)', 0.0, 1.0, 0.0)
    
    def predict_user_rating(user_id, movie_id, user_similarity_df, ratings, global_avg):
        if user_id not in user_similarity_df.index:
            movie_row = movies[movies['movieId'] == movie_id]
            return movie_row['weighted_rating'].iloc[0] if not movie_row.empty else global_avg
        
        user_idx = user_similarity_df.index.get_loc(user_id)
        similar_users = user_similarity_df.iloc[user_idx].sort_values(ascending=False).iloc[1:11]
        
        total_weight = 0
        weighted_sum = 0
        user_avg = ratings[ratings['userId'] == user_id]['rating'].mean() or global_avg
        
        for sim_user_id, similarity in similar_users.items():
            if similarity < 0.2:  # Stricter threshold for better personalization
                break
            user_ratings = ratings[(ratings['userId'] == sim_user_id) & (ratings['movieId'] == movie_id)]
            if not user_ratings.empty:
                rating = user_ratings['rating'].iloc[0]
                sim_user_avg = ratings[ratings['userId'] == sim_user_id]['rating'].mean() or global_avg
                adjusted_rating = rating - sim_user_avg + user_avg
                weighted_sum += similarity * adjusted_rating
                total_weight += np.abs(similarity)
        
        if total_weight > 0:
            prediction = weighted_sum / total_weight
            return np.clip(prediction, 0.5, 5.0)
        else:
            movie_row = movies[movies['movieId'] == movie_id]
            return movie_row['weighted_rating'].iloc[0] if not movie_row.empty else global_avg
    
    def recommend_similar_movies(query_movie_title, user_id=None, top_n=10, diversity_weight=0.0):
        query_movie = movies[movies['title'] == query_movie_title]
        if query_movie.empty:
            return "Movie not found."
        
        query_movie_id = query_movie['movieId'].iloc[0]
        if query_movie_id not in tfidf_matrix.index:
            return "Movie not found in TF-IDF matrix."
        
        query_vector = tfidf_matrix.loc[query_movie_id].values
        query_genres = set(query_movie['genres_list'].iloc[0])
        
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
        
        if user_id and user_id != "":
            try:
                user_id_int = int(user_id)
                user_ratings = ratings[ratings['userId'] == user_id_int]
                seen_movies = user_ratings['movieId'].unique()
                candidates = candidates[~candidates['movieId'].isin(seen_movies)]
                
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
        
        # Normalize similarity
        candidates['similarity'] = candidates['similarity'] / candidates['similarity'].max() if candidates['similarity'].max() > 0 else 1.0
        candidates['quality_score'] = candidates['predicted_rating']
        candidates['combined_score'] = candidates['similarity'] * candidates['quality_score']
        
        # Apply diversity adjustment if diversity_weight > 0
        if diversity_weight > 0:
            top_candidates = candidates.sort_values('combined_score', ascending=False).head(top_n * 2)
            selected_movies = []
            remaining_candidates = top_candidates.copy()
            
            while len(selected_movies) < top_n and not remaining_candidates.empty:
                top_movie = remaining_candidates.iloc[0]
                selected_movies.append(top_movie)
                remaining_candidates = remaining_candidates.iloc[1:]
                
                if len(selected_movies) > 1:
                    selected_ids = [m['movieId'] for m in selected_movies]
                    div_scores = []
                    for _, cand in remaining_candidates.iterrows():
                        cand_vector = tfidf_matrix.loc[cand['movieId']].values
                        avg_dist = np.mean([cosine(cand_vector, tfidf_matrix.loc[sid].values) for sid in selected_ids])
                        div_scores.append(avg_dist)
                    remaining_candidates['diversity_score'] = div_scores
                    remaining_candidates['combined_score'] = (
                        (1 - diversity_weight) * remaining_candidates['combined_score'] +
                        diversity_weight * remaining_candidates['diversity_score'] * remaining_candidates['quality_score']
                    )
                    remaining_candidates = remaining_candidates.sort_values('combined_score', ascending=False)
            top_recs = pd.DataFrame(selected_movies).head(top_n)
        else:
            top_recs = candidates.sort_values('combined_score', ascending=False).head(top_n)
        
        results = []
        for _, row in top_recs.iterrows():
            shared_genres = query_genres.intersection(set(row['genres_list']))
            sim = row['similarity'] if not np.isnan(row['similarity']) else 0.0
            pred = row['predicted_rating'] if not np.isnan(row['predicted_rating']) else global_avg
            explanation = (
                f"Similarity: {sim:.2f} | "
                f"{'Predicted' if user_id else 'Weighted'} Rating: {pred:.2f} | "
                f"Shared Genres: {', '.join(list(shared_genres)[:3]) or 'None'}"
            )
            results.append((row['title'], explanation, pred, row['combined_score']))
        
        return results
    
    if st.button("🎯 Get Personalized Recommendations"):
        with st.spinner("Computing personalized recommendations..."):
            recs = recommend_similar_movies(selected_movie, user_id_input, top_n, diversity_weight)
            
            if isinstance(recs, str):
                st.error(recs)
            elif len(recs) == 0:
                st.error("No recommendations available with current filters.")
            else:
                st.subheader(f"📺 Top {len(recs)} Recommendations for '{selected_movie}'")
                
                if user_id_input and user_id_input != "":
                    try:
                        user_id_int = int(user_id_input)
                        user_rating_count = len(ratings[ratings['userId'] == user_id_int])
                        st.metric("User's Total Ratings", user_rating_count)
                        st.info(f"🔍 Personalized using User {user_id_int}'s rating history and similar users")
                    except ValueError:
                        st.info("🌐 Content-based recommendations (no user personalization)")
                else:
                    st.info("🌐 Content-based recommendations (no user personalization)")
                
                for i, (title, explanation, pred_rating, score) in enumerate(recs, 1):
                    with st.expander(f"{i}. {title} (Score: {score:.3f})"):
                        st.write(f"**Explanation**: {explanation}")  # Ensure explanation is displayed
                        query_genres = set(movies[movies['title'] == selected_movie]['genres_list'].iloc[0])
                        rec_movie = movies[movies['title'] == title]
                        if not rec_movie.empty:
                            rec_genres = set(rec_movie['genres_list'].iloc[0])
                            shared = query_genres.intersection(rec_genres)
                            st.write(f"🎨 **Shared Genres** ({len(shared)}/{len(query_genres)}): {', '.join(list(shared)) or 'None'}")
                        else:
                            st.write("🎨 **Shared Genres**: Not available (movie not found)")
                
                if len(recs) > 0:
                    scores = [r[3] for r in recs]
                    try:
                        similarities = [float(r[1].split('Similarity: ')[1].split(' |')[0]) for r in recs]
                        avg_similarity = np.mean(similarities) if similarities else 0.0
                        st.metric("Average Similarity", f"{avg_similarity:.2f}")
                        st.metric("Recommendation Range", f"{min(scores):.3f} - {max(scores):.3f}")
                    except Exception as e:
                        st.warning(f"Error computing statistics: {e}")
                else:
                    st.warning("No recommendations available to compute statistics.")
else:
    st.warning("📁 Please upload both `movies.csv` and `ratings.csv` files to get started!")
    st.markdown("""
    ### 📋 How to get MovieLens data:
    1. Download from [GroupLens](https://grouplens.org/datasets/movielens/)
    2. Use ML-100K, ML-1M, or ML-25M datasets
    3. Upload `movies.csv` and `ratings.csv` files
    """)
