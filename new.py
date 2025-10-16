import streamlit as st
import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from collections import Counter
import urllib.request
import os
import logging
import warnings
import time
import requests
from urllib.error import URLError
import pickle
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Google Drive direct download URLs (replace with your actual links)
MOVIES_GDRIVE_URL = "https://drive.google.com/uc?export=download&id=1wZY1z-YfHAPZMRVYcx8rD8MeR4t_c_nC"
RATINGS_GDRIVE_URL = "https://drive.google.com/uc?export=download&id=1pR3LYyvl7kUIJ0R0KkGoOYN8DKhNOIFJ"
# Optional: Precomputed similarity matrix (upload to Google Drive if used)
SIMILARITY_MATRIX_URL = "https://drive.google.com/uc?export=download&id=YOUR_SIMILARITY_MATRIX_FILE_ID"

class EnhancedMovieRecommender:
    def __init__(self, movies_url=MOVIES_GDRIVE_URL, ratings_url=RATINGS_GDRIVE_URL, similarity_url=None):
        """Initialize the recommender system with data from Google Drive URLs."""
        self.movies_file = "movies_temp.csv"
        self.ratings_file = "ratings_temp.csv"
        self.similarity_file = "similarity_matrix.pkl"
        
        # Progress tracking
        self.progress_steps = [
            "Downloading movies data",
            "Downloading ratings data",
            "Loading movies data",
            "Loading ratings data",
            "Preprocessing genres",
            "Computing rating stats",
            "Building similarity matrix",
            "Building user-based model"
        ]
        self.current_step = 0
        
        try:
            # Download files with timeout
            logger.info(self.progress_steps[0])
            st.write(self.progress_steps[0])
            self._download_with_timeout(movies_url, self.movies_file)
            logger.info(self.progress_steps[1])
            st.write(self.progress_steps[1])
            self._download_with_timeout(ratings_url, self.ratings_file)
            
            # Load data
            logger.info(self.progress_steps[2])
            st.write(self.progress_steps[2])
            self.movies = pd.read_csv(self.movies_file)
            logger.info(self.progress_steps[3])
            st.write(self.progress_steps[3])
            self.ratings = pd.read_csv(self.ratings_file)
            # Optional: Downsample ratings for large datasets
            # self.ratings = self.ratings.sample(frac=0.1, random_state=42)
        except URLError as e:
            logger.error(f"Network error during download: {e}")
            raise Exception(f"Network error downloading files: {e}")
        except FileNotFoundError as e:
            logger.error(f"File not found after download: {e}")
            raise FileNotFoundError(f"File not found after download: {e}")
        except PermissionError as e:
            logger.error(f"Permission denied: {e}")
            raise PermissionError(f"Permission denied: {e}")
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise Exception(f"Error loading data: {e}")
        
        self.global_avg = self.ratings['rating'].mean()
        self.m = 10
        logger.info(self.progress_steps[4])
        st.write(self.progress_steps[4])
        self.movies['year'] = self.movies['title'].str.extract(r'\((\d{4})\)', expand=False).astype(float)
        self._preprocess_genres()
        logger.info(self.progress_steps[5])
        st.write(self.progress_steps[5])
        self._compute_rating_stats()
        
        # Load or compute similarity matrix
        logger.info(self.progress_steps[6])
        st.write(self.progress_steps[6])
        if similarity_url:
            st.write("Loading precomputed similarity matrix...")
            self._download_with_timeout(similarity_url, self.similarity_file)
            with open(self.similarity_file, 'rb') as f:
                self.similarity_matrix = pickle.load(f)
        else:
            self._build_similarity_matrix()
        
        logger.info(self.progress_steps[7])
        st.write(self.progress_steps[7])
        self._build_user_based_model()
        
    def _download_with_timeout(self, url, output_path, timeout=30):
        """Download file with a timeout."""
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            with open(output_path, 'wb') as f:
                f.write(response.content)
        except requests.exceptions.RequestException as e:
            raise URLError(f"Failed to download {url}: {e}")
    
    def _preprocess_genres(self):
        self.movies['genres_list'] = self.movies['genres'].str.split('|')
        all_genres = [g for sublist in self.movies['genres_list'] for g in sublist]
        genre_counts = Counter(all_genres)
        N = len(self.movies)
        self.idf = {genre: np.log(N / count) for genre, count in genre_counts.items()}
        unique_genres = list(self.idf.keys())
        self.tfidf_matrix = pd.DataFrame(0.0, index=self.movies['movieId'], columns=unique_genres)
        for idx, row in self.movies.iterrows():
            for genre in row['genres_list']:
                self.tfidf_matrix.loc[row['movieId'], genre] = self.idf[genre]
        total_movies = len(self.movies)
        self.genre_importance = {genre: count / total_movies for genre, count in genre_counts.items()}
        
    def _compute_rating_stats(self):
        movie_stats = self.ratings.groupby('movieId').agg(
            avg_rating=('rating', 'mean'),
            num_ratings=('rating', 'count'),
            std_rating=('rating', 'std')
        ).reset_index()
        movie_stats['weighted_rating'] = (
            (movie_stats['num_ratings'] * movie_stats['avg_rating'] + self.m * self.global_avg) /
            (movie_stats['num_ratings'] + self.m)
        )
        movie_stats['confidence'] = 1 / (1 + movie_stats['std_rating'].fillna(0))
        movie_stats['confidence'] *= np.log1p(movie_stats['num_ratings'])
        self.movies = self.movies.merge(movie_stats, on='movieId', how='left').fillna({
            'num_ratings': 0, 'avg_rating': self.global_avg, 'std_rating': 1.0,
            'weighted_rating': self.global_avg, 'confidence': 0
        })
        
    def _build_similarity_matrix(self):
        # Limit to top N movies by number of ratings to reduce computation
        top_n_movies = 5000  # Adjust based on available memory
        if len(self.movies) > top_n_movies:
            st.write(f"Limiting similarity matrix to top {top_n_movies} movies by rating count...")
            top_movies = self.movies.sort_values('num_ratings', ascending=False).head(top_n_movies)
            self.movies = top_movies
            self.tfidf_matrix = self.tfidf_matrix.loc[top_movies['movieId']]
        
        st.write("Computing cosine similarity...")
        tfidf_array = self.tfidf_matrix.values
        cosine_sims = cosine_similarity(tfidf_array)
        self.cosine_sim_matrix = pd.DataFrame(
            cosine_sims, index=self.tfidf_matrix.index, columns=self.tfidf_matrix.index
        )
        
        st.write("Computing Jaccard similarity...")
        self.jaccard_sim_matrix = self._compute_jaccard_similarity()
        
        st.write("Combining similarity matrices...")
        self.similarity_matrix = 0.7 * self.cosine_sim_matrix + 0.3 * self.jaccard_sim_matrix
        
        # Optional: Save similarity matrix to avoid recomputation
        # with open(self.similarity_file, 'wb') as f:
        #     pickle.dump(self.similarity_matrix, f)
        
    def _compute_jaccard_similarity(self):
        jaccard_matrix = pd.DataFrame(
            np.zeros((len(self.movies), len(self.movies))), 
            index=self.movies['movieId'], 
            columns=self.movies['movieId']
        )
        total_pairs = len(self.movies) * (len(self.movies) - 1) // 2
        processed_pairs = 0
        for i, (_, movie1) in enumerate(self.movies.iterrows()):
            for j, (_, movie2) in enumerate(self.movies.iterrows()):
                if i == j:
                    jaccard_matrix.iloc[i, j] = 1.0
                    continue
                if i < j:  # Only compute upper triangle to save time
                    set1 = set(movie1['genres_list'])
                    set2 = set(movie2['genres_list'])
                    intersection = len(set1.intersection(set2))
                    union = len(set1.union(set2))
                    sim = intersection / union if union > 0 else 0
                    jaccard_matrix.iloc[i, j] = sim
                    jaccard_matrix.iloc[j, i] = sim  # Mirror to lower triangle
                    processed_pairs += 1
                    if processed_pairs % 10000 == 0:
                        st.write(f"Processed {processed_pairs}/{total_pairs} movie pairs...")
        return jaccard_matrix
    
    def _build_user_based_model(self):
        user_item_matrix = self.ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)
        svd = TruncatedSVD(n_components=20, random_state=42)  # Reduced components
        user_features = svd.fit_transform(user_item_matrix)
        self.user_similarity = cosine_similarity(user_features)
        self.user_similarity_df = pd.DataFrame(
            self.user_similarity, 
            index=user_item_matrix.index, 
            columns=user_item_matrix.index
        )
        self.item_features = pd.DataFrame(svd.components_.T, index=user_item_matrix.columns)
    
    def _predict_user_rating(self, user_id, movie_id):
        if user_id not in self.user_similarity_df.index:
            return self.movies.loc[self.movies['movieId'] == movie_id, 'weighted_rating'].iloc[0]
        user_idx = self.user_similarity_df.index.get_loc(user_id)
        similar_users = self.user_similarity_df.iloc[user_idx].sort_values(ascending=False).iloc[1:11]
        total_weight = 0
        weighted_sum = 0
        for sim_user, similarity in similar_users.items():
            if similarity < 0.1:
                break
            user_ratings = self.ratings[self.ratings['userId'] == sim_user]
            user_rating = user_ratings[user_ratings['movieId'] == movie_id]
            if not user_rating.empty:
                rating = user_rating['rating'].iloc[0]
                weighted_sum += similarity * (rating - self.global_avg)
                total_weight += similarity
        if total_weight > 0:
            prediction = self.global_avg + weighted_sum / total_weight
            return np.clip(prediction, 0.5, 5.0)
        return self.movies.loc[self.movies['movieId'] == movie_id, 'weighted_rating'].iloc[0]
    
    def _compute_diversity_score(self, movie_id, recommendations):
        query_genres = set(self.movies.loc[self.movies['movieId'] == movie_id, 'genres_list'].iloc[0])
        total_diversity = 0
        for rec_id in recommendations['movieId']:
            rec_genres = set(self.movies.loc[self.movies['movieId'] == rec_id, 'genres_list'].iloc[0])
            unique_genres = len(rec_genres - query_genres)
            total_diversity += unique_genres
        return total_diversity / len(recommendations)
    
    def recommend_similar_movies(self, query_movie_title, user_id=None, top_n=10, 
                               year_range=None, min_similarity=0.1, diversity_weight=0.2):
        query_movie = self.movies[self.movies['title'] == query_movie_title]
        if query_movie.empty:
            return "Movie not found."
        query_movie_id = query_movie['movieId'].iloc[0]
        query_genres = set(query_movie['genres_list'].iloc[0])
        similarities = self.similarity_matrix[query_movie_id].copy()
        candidates = pd.DataFrame({
            'movieId': similarities.index,
            'similarity': similarities.values,
            'content_score': similarities.values
        })
        candidates = candidates.merge(
            self.movies[['movieId', 'title', 'genres_list', 'weighted_rating', 
                        'num_ratings', 'confidence', 'year']], on='movieId'
        )
        candidates = candidates[candidates['num_ratings'] >= self.m]
        candidates = candidates[candidates['similarity'] >= min_similarity]
        if year_range:
            candidates = candidates[
                (candidates['year'].ge(year_range[0]) | candidates['year'].isna()) & 
                (candidates['year'].le(year_range[1]) | candidates['year'].isna())
            ]
        candidates = candidates[candidates['movieId'] != query_movie_id]  # Fixed syntax error
        if user_id is not None:
            user_ratings = self.ratings[self.ratings['userId'] == user_id]
            seen_movies = user_ratings['movieId'].unique()
            candidates = candidates[~candidates['movieId'].isin(seen_movies)]
            predicted_ratings = [self._predict_user_rating(user_id, row['movieId']) for _, row in candidates.iterrows()]
            candidates['predicted_rating'] = predicted_ratings
        else:
            candidates['predicted_rating'] = candidates['weighted_rating']
        candidates['quality_score'] = candidates['predicted_rating'] * candidates['confidence']
        if len(candidates) > 1:
            candidates['diversity_score'] = candidates['movieId'].apply(
                lambda x: self._compute_diversity_score(query_movie_id, candidates)
            )
        else:
            candidates['diversity_score'] = 1.0
        candidates['final_score'] = (
            candidates['similarity'] * candidates['quality_score'] * 
            (1 + diversity_weight * candidates['diversity_score'])
        )
        top_recs = candidates.sort_values('final_score', ascending=False).head(top_n)
        results = []
        for _, row in top_recs.iterrows():
            explanation = self._generate_explanation(query_movie_id, row)
            results.append({
                'title': row['title'],
                'year': row['year'],
                'similarity': row['similarity'],
                'predicted_rating': row['predicted_rating'],
                'final_score': row['final_score'],
                'explanation': explanation
            })
        return pd.DataFrame(results)
    
    def _generate_explanation(self, query_id, recommendation):
        query_genres = set(self.movies.loc[self.movies['movieId'] == query_id, 'genres_list'].iloc[0])
        rec_genres = set(recommendation['genres_list'])
        shared_genres = query_genres.intersection(rec_genres)
        important_genres = sorted(shared_genres, 
                                key=lambda g: self.genre_importance.get(g, 0), 
                                reverse=True)[:3]
        explanation_parts = []
        explanation_parts.append(f"Similarity: {recommendation['similarity']:.2f}")
        explanation_parts.append(f"Predicted Rating: {recommendation['predicted_rating']:.1f}")
        if len(important_genres) > 0:
            genre_text = ", ".join([f"{g} ({self.genre_importance[g]:.1%} rarity)" 
                                  for g in important_genres])
            explanation_parts.append(f"Key Matches: {genre_text}")
        if 'confidence' in recommendation and recommendation['confidence'] > 0.8:
            explanation_parts.append("High confidence based on many ratings")
        return " | ".join(explanation_parts)
    
    def get_movie_info(self, movie_title):
        movie = self.movies[self.movies['title'] == movie_title]
        if movie.empty:
            return None
        movie_info = movie.iloc[0]
        genres = "|".join(movie_info['genres_list'])
        return {
            'title': movie_info['title'],
            'year': movie_info['year'],
            'genres': genres,
            'avg_rating': movie_info['avg_rating'],
            'weighted_rating': movie_info['weighted_rating'],
            'num_ratings': movie_info['num_ratings'],
            'confidence': movie_info['confidence']
        }

# Streamlit Interface
def main():
    st.set_page_config(page_title="Movie Recommender System", layout="wide")
    
    # Custom CSS for styling
    st.markdown("""
        <style>
        .main { background-color: #f5f5f5; }
        .stButton>button { background-color: #4CAF50; color: white; }
        .stSelectbox { max-width: 600px; }
        .stExpander { background-color: #ffffff; border-radius: 5px; }
        .metric-card { background-color: #e8f4f8; padding: 10px; border-radius: 5px; }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("🎥 Movie Recommender System")
    st.markdown("Discover movies similar to your favorites, personalized to your taste!")
    
    # Cache the recommender
    @st.cache_resource
    def load_recommender():
        st.write("Initializing recommender...")
        recommender = EnhancedMovieRecommender(similarity_url=SIMILARITY_MATRIX_URL)
        st.write("Recommender initialized!")
        return recommender
    
    # Load recommender with error handling and progress bar
    try:
        progress_bar = st.progress(0)
        recommender = load_recommender()
        progress_bar.progress(100)
    except Exception as e:
        st.error(f"Failed to initialize recommender: {e}")
        st.markdown("**Possible fixes:**")
        st.markdown("- Ensure the Google Drive links for `movies.csv`, `ratings.csv`, and `similarity_matrix.pkl` (if used) are correct and publicly accessible.")
        st.markdown("- Check your internet connection.")
        st.markdown("- Try using a smaller dataset (e.g., MovieLens 100K).")
        st.markdown("- Clear the cache using the button below.")
        return
    
    # Sidebar for settings
    st.sidebar.header("⚙️ Settings")
    if st.sidebar.button("Clear Cache"):
        st.cache_resource.clear()
        st.success("Cache cleared! Please refresh the page.")
    
    user_id = st.sidebar.selectbox(
        "Personalize for User",
        ["None"] + sorted([f"User {uid}" for uid in recommender.ratings['userId'].unique()]),
        help="Select a user ID for personalized recommendations."
    )
    user_id = int(user_id.split()[-1]) if user_id != "None" else None
    
    year_range_options = {
        "Any Year": (None, None),
        "1990-2000": (1990, 2000),
        "2000-2010": (2000, 2010),
        "2010-2020": (2010, 2020),
        "Post-2020": (2020, None)
    }
    year_range_selection = st.sidebar.selectbox(
        "Filter by Year Range",
        list(year_range_options.keys()),
        help="Restrict recommendations to a specific time period."
    )
    year_range = year_range_options[year_range_selection]
    
    diversity_weight = st.sidebar.slider(
        "Diversity Preference",
        0.0, 1.0, 0.2, 0.1,
        help="Higher values prioritize more diverse recommendations."
    )
    
    # Main content
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.header("🔍 Find Similar Movies")
        movie_options = sorted(recommender.movies['title'].tolist())
        query_title = st.selectbox(
            "Select a movie",
            options=movie_options,
            index=movie_options.index("Iron Man (2008)") if "Iron Man (2008)" in movie_options else 0,
            help="Choose a movie to find similar recommendations."
        )
        
        if st.button("Get Recommendations", type="primary"):
            with st.spinner("Finding the best movies for you..."):
                recommendations = recommender.recommend_similar_movies(
                    query_title, user_id=user_id, top_n=10, 
                    year_range=year_range, diversity_weight=diversity_weight
                )
                
                if isinstance(recommendations, str):
                    st.error(recommendations)
                else:
                    movie_info = recommender.get_movie_info(query_title)
                    if movie_info:
                        st.subheader(f"📽️ About {query_title}")
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.metric("Average Rating", f"{movie_info['avg_rating']:.1f}/5")
                        with col_b:
                            st.metric("Weighted Rating", f"{movie_info['weighted_rating']:.1f}/5")
                        with col_c:
                            st.metric("Number of Ratings", f"{int(movie_info['num_ratings']):,}")
                        st.markdown(f"**Genres:** {movie_info['genres']}")
                        st.markdown(f"**Year:** {int(movie_info['year'])}" if not pd.isna(movie_info['year']) else "**Year:** Unknown")
                    
                    st.subheader("🎬 Top Recommendations")
                    for idx, rec in recommendations.iterrows():
                        with st.expander(f"{idx+1}. {rec['title']} ({int(rec['year']) if not pd.isna(rec['year']) else 'Unknown'})"):
                            st.markdown(f"**Score:** {rec['final_score']:.3f}")
                            st.markdown(f"**Similarity:** {rec['similarity']:.2f}")
                            st.markdown(f"**Predicted Rating:** {rec['predicted_rating']:.1f}/5")
                            st.info(rec['explanation'])
    
    with col2:
        st.header("📊 System Stats")
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Movies", f"{len(recommender.movies):,}")
        st.metric("Total Ratings", f"{len(recommender.ratings):,}")
        st.metric("Unique Users", f"{recommender.ratings['userId'].nunique():,}")
        st.metric("Average Rating", f"{recommender.global_avg:.2f}/5")
        st.markdown('</div>', unsafe_allow_html=True)
        
        genre_counts = Counter([g for sublist in recommender.movies['genres_list'] for g in sublist])
        genre_df = pd.DataFrame.from_dict(genre_counts, orient='index', columns=['count']).sort_values('count', ascending=False)
        st.subheader("📈 Genre Distribution")
        st.bar_chart(genre_df)

if __name__ == "__main__":
    main()
