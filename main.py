import random
from MusicRecommender import *

# Datayı oku ve hazırla
recommender = MusicRecommendationSystem(data_path='recommendation-dataset/data.csv')
recommender.load_data()
recommender.preprocess_data()


recommender.visualize_correlation()
# K-Means modelini eğit
recommender.train_model(n_clusters=291)

# Rastgele 60 şarkı seç
songList = random.sample(recommender.df['name'].tolist(), 60)

# Şarkı önerisi yap
recommender.recommend_songs(song_list= songList, n_recommendations=5)

recommender.visualize_clusters()

recommender.visualize_radar()