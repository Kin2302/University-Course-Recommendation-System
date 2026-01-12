import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow import keras
import os

# Set page config
st.set_page_config(page_title="Movie Recommender System", page_icon="🎬", layout="wide")

# Paths
MODEL_PATH = "best_model.keras"
USER_ENCODER_PATH = "user2user_encoded.pkl"
MOVIE_ENCODER_PATH = "movie2movie_encoded.pkl"
MOVIE_DECODER_PATH = "movie_encoded2movie.pkl"
USER_INTERACTED_PATH = "user_interacted_items.pkl"
RATINGS_PATH = "../ml-latest-small/ml-latest-small/ratings.csv"
MOVIES_PATH = "../ml-latest-small/ml-latest-small/movies.csv"


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(CURRENT_DIR, 'models')
DATA_DIR = os.path.join(CURRENT_DIR, 'data')


MODEL_PATH = os.path.join(MODELS_DIR, "best_model.keras")
USER_ENCODER_PATH = os.path.join(MODELS_DIR, "user2user_encoded.pkl")
MOVIE_ENCODER_PATH = os.path.join(MODELS_DIR, "movie2movie_encoded.pkl")
MOVIE_DECODER_PATH = os.path.join(MODELS_DIR, "movie_encoded2movie.pkl")
USER_INTERACTED_PATH = os.path.join(MODELS_DIR, "user_interacted_items.pkl"


RATINGS_PATH = os.path.join(DATA_DIR, "ratings.csv") 
MOVIES_PATH = os.path.join(DATA_DIR, "movies.csv")


st.set_page_config(page_title="Movie Recommender System", page_icon="🎬", layout="wide")
@st.cache_resource
def load_model_and_data():
    """Load the trained model and all necessary data files"""
    # Load model
    model = keras.models.load_model(MODEL_PATH)
    
    # Load pickle files
    with open(USER_ENCODER_PATH, 'rb') as f:
        user2user_encoded = pickle.load(f)
    
    with open(MOVIE_ENCODER_PATH, 'rb') as f:
        movie2movie_encoded = pickle.load(f)
    
    with open(MOVIE_DECODER_PATH, 'rb') as f:
        movie_encoded2movie = pickle.load(f)
    
    with open(USER_INTERACTED_PATH, 'rb') as f:
        user_interacted_items = pickle.load(f)
    
    # Load CSV files
    ratings_df = pd.read_csv(RATINGS_PATH)
    movies_df = pd.read_csv(MOVIES_PATH)
    
    return model, user2user_encoded, movie2movie_encoded, movie_encoded2movie, user_interacted_items, ratings_df, movies_df

# Load all data
try:
    model, user2user_encoded, movie2movie_encoded, movie_encoded2movie, user_interacted_items, ratings_df, movies_df = load_model_and_data()
    
    # Get valid user IDs
    valid_user_ids = sorted(ratings_df['userId'].unique())
    min_user_id = min(valid_user_ids)
    max_user_id = max(valid_user_ids)
    
except Exception as e:
    st.error(f"Error loading model or data: {e}")
    st.stop()

# App title and description
st.title("🎬 Movie Recommender System")
st.markdown("### Personalized Movie Recommendations Based on Your Viewing History")
st.markdown("---")

# Create tabs for different recommendation modes
tab1, tab2 = st.tabs(["🎯 Existing User", "🆕 New User (Cold Start)"])

# TAB 1: Existing User Recommendations
with tab1:
    st.markdown("### Get recommendations for existing users in the system")
    
    # User input section
    col1, col2 = st.columns([1, 3])
    with col1:
        user_id = st.number_input(
            f"Enter User ID ({min_user_id}-{max_user_id}):",
            min_value=int(min_user_id),
            max_value=int(max_user_id),
            value=int(min_user_id),
            step=1,
            key="existing_user_id"
        )

    with col2:
        st.markdown("")
        st.markdown("")
        if st.button("Get Recommendations", type="primary", key="existing_user_btn"):
            st.session_state['show_recommendations'] = True
        else:
            if 'show_recommendations' not in st.session_state:
                st.session_state['show_recommendations'] = False

        st.markdown("---")

    # Validate user ID
    if user_id not in valid_user_ids:
        st.error(f"❌ Invalid User ID! Please enter a user ID between {min_user_id} and {max_user_id}.")
        st.stop()

    # Only show results if button was clicked or already showing
    if st.session_state.get('show_recommendations', False):
        
        # Get user's watched movies
        user_ratings = ratings_df[ratings_df['userId'] == user_id].copy()
        user_watched = user_ratings.merge(movies_df, on='movieId', how='left')
        user_watched = user_watched.sort_values('rating', ascending=False)
        
        # Display watched movies
        st.subheader(f"🎥 Movies Watched by User {user_id}")
        st.caption(f"Total movies watched: {len(user_watched)}")
        
        # Format the watched movies dataframe
        watched_display = user_watched[['title', 'genres', 'rating']].copy()
        watched_display.columns = ['Movie Title', 'Genres', 'Your Rating']
        watched_display['Your Rating'] = watched_display['Your Rating'].apply(lambda x: f"⭐ {x:.1f}")
        watched_display.index = range(1, len(watched_display) + 1)
        
        st.dataframe(watched_display, use_container_width=True, height=300)
        
        # Analyze and display favorite genres
        st.markdown("### 📊 Your Favorite Genres")
        
        # Extract all genres from watched movies
        all_genres = []
        for genres_str in user_watched['genres'].dropna():
            genres_list = genres_str.split('|')
            all_genres.extend(genres_list)
        
        # Count genre occurrences
        if all_genres:
            genre_counts = pd.Series(all_genres).value_counts()
            
            # Display top genres
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Create a horizontal bar chart
                top_genres = genre_counts.head(10)
                chart_data = pd.DataFrame({
                    'Genre': top_genres.index,
                    'Count': top_genres.values
                })
                st.bar_chart(chart_data.set_index('Genre'), height=250)
            
            with col2:
                st.markdown("**Top Genres:**")
                for i, (genre, count) in enumerate(genre_counts.head(10).items(), 1):
                    percentage = (count / len(user_watched)) * 100
                    st.write(f"{i}. **{genre}**: {count} movies ({percentage:.1f}%)")
        
        st.markdown("---")
        
        # Generate recommendations
        st.subheader("🎯 Recommended Movies for You")
        
        with st.spinner("Generating personalized recommendations..."):
            try:
                # Encode user ID
                user_encoded = user2user_encoded.get(user_id)
                
                if user_encoded is None:
                    st.error(f"User ID {user_id} not found in encoding mapping.")
                    st.stop()
                
                # Get movies the user has already watched
                watched_movie_ids = set(user_watched['movieId'].values)
                
                # Get all movies
                all_movie_ids = movies_df['movieId'].values
                
                # Get unwatched movies
                unwatched_movie_ids = [mid for mid in all_movie_ids if mid not in watched_movie_ids]
                
                # Encode unwatched movies
                unwatched_encoded = []
                unwatched_original = []
                
                for movie_id in unwatched_movie_ids:
                    encoded_id = movie2movie_encoded.get(movie_id)
                    if encoded_id is not None:
                        unwatched_encoded.append(encoded_id)
                        unwatched_original.append(movie_id)
                
                if len(unwatched_encoded) == 0:
                    st.warning("No unwatched movies available for recommendations.")
                    st.stop()
                
                # Prepare input for model
                user_array = np.array([user_encoded] * len(unwatched_encoded))
                movie_array = np.array(unwatched_encoded)
                
                # Predict ratings
                predictions = model.predict([user_array, movie_array], verbose=0)
                predictions = predictions.flatten()
                
                # Create recommendations dataframe
                recommendations_df = pd.DataFrame({
                    'movieId': unwatched_original,
                    'predicted_rating': predictions
                })
                
                # Merge with movie details
                recommendations_df = recommendations_df.merge(movies_df, on='movieId', how='left')
                
                # Sort by predicted rating
                recommendations_df = recommendations_df.sort_values('predicted_rating', ascending=False)
                
                # Get top 10
                top_recommendations = recommendations_df.head(10)
                
                # Display recommendations
                st.caption("Top 10 movies you might enjoy:")
                
                # Format the recommendations dataframe
                rec_display = top_recommendations[['title', 'genres', 'predicted_rating']].copy()
                rec_display.columns = ['Movie Title', 'Genres', 'Predicted Rating']
                rec_display['Predicted Rating'] = rec_display['Predicted Rating'].apply(lambda x: f"⭐ {x:.2f}")
                rec_display.index = range(1, len(rec_display) + 1)
                
                st.dataframe(rec_display, use_container_width=True, height=400)
                
                # Analyze and display recommended movie genres
                st.markdown("---")
                st.markdown("### 📊 Genre Distribution in Recommended Movies")
                
                # Extract all genres from recommended movies
                recommended_genres = []
                for genres_str in top_recommendations['genres'].dropna():
                    genres_list = genres_str.split('|')
                    recommended_genres.extend(genres_list)
                
                # Count genre occurrences
                if recommended_genres:
                    rec_genre_counts = pd.Series(recommended_genres).value_counts()
                    
                    # Display genre statistics
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        # Create a horizontal bar chart
                        chart_data = pd.DataFrame({
                            'Genre': rec_genre_counts.index,
                            'Count': rec_genre_counts.values
                        })
                        st.bar_chart(chart_data.set_index('Genre'), height=250)
                    
                    with col2:
                        st.markdown("**Top Genres in Recommendations:**")
                        for i, (genre, count) in enumerate(rec_genre_counts.items(), 1):
                            percentage = (count / len(top_recommendations)) * 100
                            st.write(f"{i}. **{genre}**: {count} movies ({percentage:.1f}%)")
                else:
                    st.warning("No genre information available for recommended movies.")
                
                # Show some stats
                st.markdown("---")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Movies Watched", len(user_watched))
                with col2:
                    st.metric("Average Rating Given", f"{user_watched['rating'].mean():.2f}")
                with col3:
                    st.metric("Movies Available", len(unwatched_original))
                    
            except Exception as e:
                st.error(f"Error generating recommendations: {e}")
                st.exception(e)
    else:
        st.info("👆 Enter your User ID and click 'Get Recommendations' to see personalized movie suggestions!")

# TAB 2: New User (Cold Start) Recommendations
with tab2:
    st.markdown("### Test recommendations for a new user not in the system")
    st.info("🆕 This simulates a 'cold start' scenario where the user has no history. The system will use popular movies and genre preferences.")
    
    # Genre selection
    all_unique_genres = set()
    for genres_str in movies_df['genres'].dropna():
        genres_list = genres_str.split('|')
        all_unique_genres.update(genres_list)
    all_unique_genres = sorted(list(all_unique_genres))
    
    # Remove (no genres listed) from the options
    if '(no genres listed)' in all_unique_genres:
        all_unique_genres.remove('(no genres listed)')
    
    st.markdown("#### Select your favorite genres:")
    selected_genres = st.multiselect(
        "Choose at least one genre you like:",
        options=all_unique_genres,
        default=['Action', 'Adventure'] if 'Action' in all_unique_genres and 'Adventure' in all_unique_genres else all_unique_genres[:2],
        key="new_user_genres"
    )
    
    # Number of recommendations
    num_recommendations = st.slider(
        "Number of recommendations:",
        min_value=5,
        max_value=20,
        value=10,
        key="new_user_num_recs"
    )
    
    # Strategy selection
    recommendation_strategy = st.radio(
        "Recommendation strategy:",
        options=[
            "Popular movies in selected genres",
            "Highly rated movies in selected genres",
            "Recent movies in selected genres"
        ],
        key="new_user_strategy"
    )
    
    if st.button("Get Recommendations for New User", type="primary", key="new_user_btn"):
        if len(selected_genres) == 0:
            st.error("❌ Please select at least one genre!")
        else:
            with st.spinner("Generating recommendations for new user..."):
                try:
                    # Filter movies by selected genres
                    filtered_movies = movies_df[
                        movies_df['genres'].apply(
                            lambda x: any(genre in x for genre in selected_genres) if pd.notna(x) else False
                        )
                    ].copy()
                    
                    if len(filtered_movies) == 0:
                        st.warning("No movies found for the selected genres.")
                    else:
                        # Merge with ratings to get popularity and average ratings
                        movie_stats = ratings_df.groupby('movieId').agg({
                            'rating': ['mean', 'count'],
                            'timestamp': 'max'
                        }).reset_index()
                        movie_stats.columns = ['movieId', 'avg_rating', 'rating_count', 'latest_timestamp']
                        
                        filtered_movies = filtered_movies.merge(movie_stats, on='movieId', how='left')
                        filtered_movies['avg_rating'] = filtered_movies['avg_rating'].fillna(0)
                        filtered_movies['rating_count'] = filtered_movies['rating_count'].fillna(0)
                        
                        # Apply selected strategy
                        if recommendation_strategy == "Popular movies in selected genres":
                            # Sort by number of ratings (popularity)
                            filtered_movies = filtered_movies.sort_values('rating_count', ascending=False)
                            strategy_desc = "most popular"
                        elif recommendation_strategy == "Highly rated movies in selected genres":
                            # Filter movies with at least 10 ratings and sort by average rating
                            filtered_movies = filtered_movies[filtered_movies['rating_count'] >= 10]
                            filtered_movies = filtered_movies.sort_values('avg_rating', ascending=False)
                            strategy_desc = "highest rated"
                        else:  # Recent movies
                            # Sort by latest timestamp
                            filtered_movies = filtered_movies.sort_values('latest_timestamp', ascending=False)
                            strategy_desc = "most recent"
                        
                        # Get top recommendations
                        top_recommendations = filtered_movies.head(num_recommendations)
                        
                        # Display results
                        st.success(f"✅ Found {len(filtered_movies)} movies matching your preferences!")
                        st.subheader(f"🎯 Top {num_recommendations} {strategy_desc} movies for you")
                        st.caption(f"Based on your selected genres: {', '.join(selected_genres)}")
                        
                        # Format recommendations
                        rec_display = top_recommendations[['title', 'genres', 'avg_rating', 'rating_count']].copy()
                        rec_display.columns = ['Movie Title', 'Genres', 'Avg Rating', 'Number of Ratings']
                        rec_display['Avg Rating'] = rec_display['Avg Rating'].apply(lambda x: f"⭐ {x:.2f}" if x > 0 else "N/A")
                        rec_display['Number of Ratings'] = rec_display['Number of Ratings'].apply(lambda x: f"{int(x):,}")
                        rec_display.index = range(1, len(rec_display) + 1)
                        
                        st.dataframe(rec_display, use_container_width=True, height=400)
                        
                        # Show statistics
                        st.markdown("---")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Genres Selected", len(selected_genres))
                        with col2:
                            st.metric("Matching Movies", len(filtered_movies))
                        with col3:
                            avg_of_displayed = top_recommendations['avg_rating'].mean()
                            st.metric("Average Rating", f"{avg_of_displayed:.2f}" if avg_of_displayed > 0 else "N/A")
                        
                except Exception as e:
                    st.error(f"Error generating recommendations: {e}")
                    st.exception(e)
    else:
        st.info("👆 Select your favorite genres and click 'Get Recommendations for New User' to see suggestions!")
