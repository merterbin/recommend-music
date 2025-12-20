import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class MusicRecommendationSystem:
    def __init__(self, data_path, genre_path):
        self.data_path = data_path
        self.genre_path = genre_path
        self.df = None
        self.scaler = StandardScaler()
        self.kmeans = None
        self.pca = PCA(n_components=2) 

        self.user_centroid_scaled = None 
        
        self.feature_cols = [
            'acousticness', 'danceability', 'energy', 'instrumentalness', 
            'liveness', 'loudness', 'speechiness', 'tempo', 'valence', 'popularity'
        ]
        
    def load_data(self):
        """Verileri yükler ve temizler."""
        try:
            self.df = pd.read_csv(self.data_path)

            self.df.dropna(subset=self.feature_cols, inplace=True)
            print(f"[+] Veriler yüklendi. Toplam Şarkı: {len(self.df)}")
        except FileNotFoundError:
            print("[!] Dosya bulunamadı.")

    def preprocess_data(self):
        """Veriyi normalize eder."""
        print("[-] Veri standardizasyonu yapılıyor...")
        X = self.df[self.feature_cols]
        self.scaled_features = self.scaler.fit_transform(X)
        print("[+] Standardizasyon tamamlandı.")

    def train_model(self, n_clusters=10):

        print(f"[-] {n_clusters} küme ile eğitim başladı...")
        self.kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, max_iter=300, random_state=42)
        self.kmeans.fit(self.scaled_features)
        self.df['cluster'] = self.kmeans.labels_
        print("[+] Model eğitildi.")

    def print_features_stats(self, features_series, title="Özellikler"):

        print(f"\n--- {title} ---")

        if isinstance(features_series, pd.DataFrame):
            data = features_series.iloc[0].to_dict()
        else:
            data = features_series.to_dict()

        for key in self.feature_cols:
            if key in data:
                print(f"{key.capitalize():<16}: {data[key]:.4f}")
        print("-" * 30)

    def recommend_songs(self, song_list, n_recommendations=5):

        song_list_lower = [s.lower() for s in song_list]
        playlist_df = self.df[self.df['name'].str.lower().isin(song_list_lower)]
        
        if len(playlist_df) == 0:
            print("[!] Şarkılar bulunamadı.")
            return

        print(f"\n[+] Analiz edilen şarkı sayısı: {len(playlist_df)}")

        user_profile_vector = playlist_df[self.feature_cols].mean(axis=0)

        user_profile_df = pd.DataFrame([user_profile_vector.values], columns=self.feature_cols)
        user_profile_scaled = self.scaler.transform(user_profile_df)
        
 
        self.print_features_comparison(user_profile_vector, user_profile_scaled)
    
        self.user_centroid_scaled = user_profile_scaled

        predicted_cluster = self.kmeans.predict(self.user_centroid_scaled)[0]

        cluster_size = len(self.df[self.df['cluster'] == predicted_cluster])
        print(f"[-] Tarzınız 'Cluster {predicted_cluster}' grubuna ait.")
        print(f"[-] Bu kümede toplam {cluster_size} adet şarkı var.") 

        recommendations = self.df[self.df['cluster'] == predicted_cluster]
        recommendations = recommendations[~recommendations['name'].str.lower().isin(song_list_lower)]
        
        if not recommendations.empty:
            rec_songs = recommendations.sample(n=min(n_recommendations, len(recommendations)))
            print("\n--- ÖNERİLEN ŞARKILAR ---")
            print(rec_songs[['name', 'artists', 'popularity']].to_string(index=False))
        else:
            print("[!] Bu kümede size önerecek başka şarkı kalmadı (Listeniz kümenin tamamını kapsıyor olabilir).")

    def print_features_comparison(self, raw_vector, scaled_vector):

        print(f"\n{'='*20} VERİ MANİPÜLASYONU KONTROLÜ {'='*20}")
        print(f"{'ÖZELLİK':<18} | {'HAM DEĞER (İnsan)':<18} | {'ÖLÇEKLENMİŞ (Makine)'}")
        print("-" * 65)
        
        scaled_flat = scaled_vector.flatten()
      
        raw_dict = raw_vector.to_dict()
        
        for i, col in enumerate(self.feature_cols):
            raw_val = raw_dict[col]
            scaled_val = scaled_flat[i]
            print(f"{col.capitalize():<18} | {raw_val:>10.4f}         | {scaled_val:>10.4f}")
        print("-" * 65)
        print("Not: Makine, K-Means hesaplamasında SAĞDAKİ değerleri kullanır.\n")

    def visualize_clusters(self):

        print("\n[-] Görselleştirme hazırlanıyor...")

        principal_components = self.pca.fit_transform(self.scaled_features)

        pc_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
        pc_df['cluster'] = self.df['cluster']
        
        plt.figure(figsize=(12, 8))
     
        sns.scatterplot(x='PC1', y='PC2', hue='cluster', data=pc_df, palette='viridis', alpha=0.5, s=40)

        if self.user_centroid_scaled is not None:
            user_pca = self.pca.transform(self.user_centroid_scaled)
            
            plt.scatter(
                user_pca[0, 0], user_pca[0, 1], 
                c='red', s=300, marker='X', edgecolors='black', linewidth=2,
                label='SİZİN KONUMUNUZ'
            )
            print(f"[+] Sizin konumunuz haritaya eklendi: X={user_pca[0,0]:.2f}, Y={user_pca[0,1]:.2f}")

        plt.title('Müzik Kümeleri ve Sizin Konumunuz')
        plt.xlabel('Principal Component 1 (Genel Varyans)')
        plt.ylabel('Principal Component 2')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()