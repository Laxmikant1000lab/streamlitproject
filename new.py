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
        return EnhancedMovieRecommender()
    
    recommender = load_recommender()
    
    # Sidebar for settings
    st.sidebar.header("⚙️ Settings")
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
                    # Display movie info
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
                        st.markdown(f"**Year:** {int(movie_info['year'])}")
                    
                    # Display recommendations
                    st.subheader("🎬 Top Recommendations")
                    for idx, rec in recommendations.iterrows():
                        with st.expander(f"{idx+1}. {rec['title']} ({int(rec['year'])})"):
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
        
        # Genre distribution chart
        genre_counts = Counter([g for sublist in recommender.movies['genres_list'] for g in sublist])
        genre_df = pd.DataFrame.from_dict(genre_counts, orient='index', columns=['count']).sort_values('count', ascending=False)
        st.subheader("📈 Genre Distribution")
        st.bar_chart(genre_df)

if __name__ == "__main__":
    main()
