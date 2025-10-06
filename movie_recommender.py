import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine
import streamlit as st
import matplotlib.pyplot as plt
from textblob import TextBlob  # For sentiment analysis
from sklearn.cluster import KMeans  # For user clustering
import io
import base64

# Step 1: Data Setup
data = {
    'Inception': [5, 4, 3, 5, np.nan],
    'The Matrix': [4, 5, 3, np.nan, 4],
    'Interstellar': [5, 4, np.nan, 5, 3],
    'Toy Story': [2, 1, 4, 2, np.nan],
    'The Godfather': [3, 2, 5, 3, 2],
    'Frozen': [1, np.nan, 5, 1, np.nan]
}
users = ['Alice', 'Bob', 'Carol', 'David', 'Eve']
ratings = pd.DataFrame(data, index=users)

# Simulated user comments (in practice, scrape or collect from a dataset)
comments = {
    'Alice': {'Inception': 'Mind-blowing plot!', 'The Matrix': 'Loved the action!'},
    'Bob': {'The Matrix': 'Epic sci-fi!', 'Toy Story': 'Not my style.'},
    'Carol': {'The Godfather': 'A classic masterpiece!', 'Frozen': 'So heartwarming!'},
    'David': {'Interstellar': 'Deep and emotional.', 'Frozen': 'Boring.'},
    'Eve': {'The Matrix': 'Super cool effects!'}
}

# Movie genres for content-based filtering
genres = {
    'Inception': ['Sci-Fi', 'Thriller'],
    'The Matrix': ['Sci-Fi', 'Action'],
    'Interstellar': ['Sci-Fi', 'Drama'],
    'Toy Story': ['Animation', 'Family'],
    'The Godfather': ['Crime', 'Drama'],
    'Frozen': ['Animation', 'Family']
}

# Step 2: Sentiment Analysis for Content-Based Boost
def get_sentiment_score(comment):
    if not comment:
        return 0
    analysis = TextBlob(comment)
    return analysis.sentiment.polarity  # -1 (negative) to 1 (positive)

# Step 3: User Similarity (Collaborative Filtering)
def user_similarity(u1, u2):
    common = ratings.loc[u1].notna() & ratings.loc[u2].notna()
    if common.sum() < 2:
        return 0
    vec1 = ratings.loc[u1][common]
    vec2 = ratings.loc[u2][common]
    return 1 - cosine(vec1, vec2)

# Compute similarity matrix
sim_matrix = pd.DataFrame(index=users, columns=users)
for u1 in users:
    for u2 in users:
        sim_matrix.loc[u1, u2] = user_similarity(u1, u2)

# Step 4: Predict Rating with Hybrid Approach
def predict_rating(user, movie, weight_sentiment=0.3):
    # Collaborative filtering component
    rated_users = ratings[movie].dropna().index
    if len(rated_users) == 0:
        return np.nan
    sims = sim_matrix.loc[user, rated_users]
    rates = ratings.loc[rated_users, movie]
    total_sim = sims.abs().sum()
    collab_score = (sims * rates).sum() / total_sim if total_sim > 0 else np.nan
    
    # Content-based component (sentiment + genre match)
    sentiment_score = 0
    genre_score = 0
    if user in comments and movie in comments[user]:
        sentiment_score = get_sentiment_score(comments[user][movie])
    # Genre similarity (simple: count matching genres with rated movies)
    user_rated = ratings.loc[user].dropna().index
    user_genres = set()
    for m in user_rated:
        user_genres.update(genres[m])
    common_genres = len(set(genres[movie]).intersection(user_genres))
    genre_score = common_genres / len(genres[movie]) if genres[movie] else 0
    
    # Combine scores
    return collab_score * (1 - weight_sentiment) + (sentiment_score + genre_score) * weight_sentiment

# Step 5: User Clustering for Visualization
def cluster_users(n_clusters=2):
    # Fill NaNs with 0 for clustering
    ratings_filled = ratings.fillna(0)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(ratings_filled)
    return clusters

# Step 6: Generate Explanation
def generate_explanation(user, movie, predicted_score):
    similar_users = sim_matrix.loc[user].sort_values(ascending=False).index[1:3]
    reasons = [f"Your taste is similar to {u} (similarity: {sim_matrix.loc[user, u]:.2f})" for u in similar_users]
    if user in comments and movie in comments[user]:
        sentiment = get_sentiment_score(comments[user][movie])
        reasons.append(f"Your comment on {movie} was {'positive' if sentiment > 0 else 'negative'}.")
    user_rated = ratings.loc[user].dropna().index
    user_genres = set()
    for m in user_rated:
        user_genres.update(genres[m])
    if set(genres[movie]).intersection(user_genres):
        reasons.append(f"{movie} matches genres you like: {', '.join(set(genres[movie]).intersection(user_genres))}")
    return f"Recommended {movie} (Score: {predicted_score:.2f}) because: " + "; ".join(reasons)

# Step 7: Streamlit App
def main():
    st.title("🎬 Standout Movie Recommender")
    st.write("Get personalized movie recommendations with explanations!")

    # User selection or new user input
    user = st.selectbox("Select or enter your username", users + ["New User"])
    if user == "New User":
        user = st.text_input("Enter your username", "NewUser")
        ratings[user] = pd.Series(index=ratings.columns, dtype=float)
        comments[user] = {}

    # Input ratings
    st.subheader("Rate some movies (1-5)")
    for movie in ratings.columns:
        rating = st.slider(f"{movie}", 0, 5, 0, key=f"{user}_{movie}")
        if rating > 0:
            ratings.loc[user, movie] = rating
        comment = st.text_input(f"Comment on {movie} (optional)", key=f"comment_{movie}")
        if comment:
            comments[user][movie] = comment

    # Update similarity matrix for new user
    if user not in sim_matrix.index:
        sim_matrix[user] = 0
        sim_matrix.loc[user] = 0
        for u in sim_matrix.index:
            sim_matrix.loc[user, u] = user_similarity(user, u)
            sim_matrix.loc[u, user] = sim_matrix.loc[user, u]

    # Generate recommendations
    if st.button("Get Recommendations"):
        unrated_movies = ratings.loc[user][ratings.loc[user].isna()].index
        predictions = {movie: predict_rating(user, movie) for movie in unrated_movies}
        top_recs = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:3]
        
        st.subheader("Your Top Recommendations")
        for movie, score in top_recs:
            explanation = generate_explanation(user, movie, score)
            st.write(explanation)

        # Visualize user clusters
        st.subheader("User Preference Clusters")
        clusters = cluster_users()
        ratings_filled = ratings.fillna(0)
        plt.scatter(ratings_filled.iloc[:, 0], ratings_filled.iloc[:, 1], c=clusters, cmap='viridis')
        plt.xlabel(ratings.columns[0])
        plt.ylabel(ratings.columns[1])
        plt.title("User Clusters by Preferences")
        for i, user_name in enumerate(ratings.index):
            plt.annotate(user_name, (ratings_filled.iloc[i, 0], ratings_filled.iloc[i, 1]))
        # Save plot to display in Streamlit
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_str = base64.b64encode(buf.getvalue()).decode()
        st.image(f"data:image/png;base64,{img_str}")
        plt.close()

if __name__ == "__main__":
    main()
