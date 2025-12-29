import random
from MusicRecommender import *

recommender = MusicRecommendationSystem(data_path='recommendation-dataset/data.csv', genre_path='recommendation-dataset/data_by_genres.csv')
recommender.load_data()
recommender.preprocess_data()


recommender.visualize_correlation()

recommender.train_model(n_clusters=300)

songList = random.sample(recommender.df['name'].tolist(), 60)


print("\n--- Playlist Analizi Başlıyor ---")
recommender.recommend_songs(song_list= songList, n_recommendations=5)

recommender.visualize_clusters()

recommender.visualize_radar()