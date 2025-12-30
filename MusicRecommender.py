import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class MusicRecommendationSystem:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.scaler = StandardScaler()
        self.kmeans = None
        self.pca = PCA(n_components=2) 

        self.user_centroid_scaled = None 
        self.user_profile_vector = None 
        self.last_recommendations = None
        
        self.feature_cols = [
            'acousticness', 'danceability', 'energy', 'instrumentalness', 
            'liveness', 'loudness', 'speechiness', 'tempo', 'valence', 'popularity'
        ]

    # Datayı yükle ve eksik verileri temizle
    def load_data(self):
        try:
            self.df = pd.read_csv(self.data_path)
            # Eksik verileri temizle (Sadece özellik sütunlarında)
            self.df.dropna(subset=self.feature_cols, inplace=True)
            print(f"Veriler yüklendi. Toplam Şarkı: {len(self.df)}")
        except FileNotFoundError:
            print("Dosya bulunamadı.")

    # Veriyi 0-1 aralığında standardize et
    def preprocess_data(self):
        print("Veri standardizasyonu yapılıyor...")
        X = self.df[self.feature_cols]
        self.scaled_features = self.scaler.fit_transform(X)
        print("Standardizasyon tamamlandı.")

    # K- Means modelini eğit
    def train_model(self, n_clusters=291):
        print(f"{n_clusters} küme ile eğitim başladı...")
        self.kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, max_iter=300, random_state=42)
        self.kmeans.fit(self.scaled_features)
        # Küme etiketlerini orijinal dataframe'e ekle
        self.df['cluster'] = self.kmeans.labels_
        print("Model eğitildi.")

    # Şarkı önerisi yap
    def recommend_songs(self, song_list, n_recommendations=20):

        # Normalize edilmiş kullanıcı profili oluştur
        song_list_lower = [s.lower() for s in song_list]
        playlist_df = self.df[self.df['name'].str.lower().isin(song_list_lower)]
        
        print(f"\nAnaliz edilen şarkı sayısı: {len(playlist_df)}")

        # Verilen şarkıların özelliklerinin medyanını al
        self.user_profile_vector = playlist_df[self.feature_cols].median(axis=0)
        # Kullanıcı profilini modelin anlayabileceği forma dönüştürür
        user_profile_df = pd.DataFrame([self.user_profile_vector.values], columns=self.feature_cols)
        user_profile_scaled = self.scaler.transform(user_profile_df)

        self.print_features_comparison(self.user_profile_vector, user_profile_scaled)
    
        self.user_centroid_scaled = user_profile_scaled
        # Küme tahmini yap
        predicted_cluster = self.kmeans.predict(self.user_centroid_scaled)[0]

        cluster_size = len(self.df[self.df['cluster'] == predicted_cluster])
        print(f"Tarzınız 'Cluster {predicted_cluster}' grubuna ait.")
        print(f"Bu kümede toplam {cluster_size} adet şarkı var.") 
        # Öneriler için aynı kümeden şarkılar seç
        recommendations = self.df[self.df['cluster'] == predicted_cluster]
        recommendations = recommendations[~recommendations['name'].str.lower().isin(song_list_lower)]
        
        if not recommendations.empty:
            # Önerilen şarkılardan rastgele seçim yap
            rec_songs = recommendations.sample(n=min(n_recommendations, len(recommendations)))
            self.last_recommendations = rec_songs
            print("\n--- ÖNERİLEN ŞARKILAR ---")
            print(rec_songs[['name', 'artists',"cluster"]].to_string(index=False))
        else:
            print("Bu kümede size önerecek başka şarkı kalmadı (Listeniz kümenin tamamını kapsıyor olabilir).")

    # Özellik karşılaştırmasını yazdır
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

    # Küme görselleştirmesi
    def visualize_clusters(self):
        if self.user_centroid_scaled is None:
            print("Hata: Önce bir şarkı listesi ile 'recommend_songs' fonksiyonunu çalıştırmalısınız.")
            return

        print("\nGelişmiş görselleştirme hazırlanıyor...")
        # PCA ile 2 boyuta indirgeme
        principal_components = self.pca.fit_transform(self.scaled_features)
        pc_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
        pc_df['cluster'] = self.df['cluster']

        # Kullanıcı merkezini PCA uzayına dönüştür
        user_pca = self.pca.transform(self.user_centroid_scaled)
        predicted_cluster = self.kmeans.predict(self.user_centroid_scaled)[0]

        plt.figure(figsize=(14, 10))


        other_clusters = pc_df[pc_df['cluster'] != predicted_cluster]
        plt.scatter(
            other_clusters['PC1'], other_clusters['PC2'], 
            c='lightgray', s=10, alpha=0.3, label='Diğer Müzikler'
        )

        # Sadece sizin kümenize ait noktaları al
        your_cluster_data = pc_df[pc_df['cluster'] == predicted_cluster]
        plt.scatter(
            your_cluster_data['PC1'], your_cluster_data['PC2'], 
            c='blue', s=50, alpha=0.8, label=f'Sizin Müzik Grubunuz (Cluster {predicted_cluster})'
        )
        #
        if self.last_recommendations is not None:
            # Önerilen şarkıların indekslerini al
            rec_indices = self.last_recommendations.index
            rec_pca = pc_df.loc[rec_indices]
            # Önerilen şarkıları vurgula
            plt.scatter(
                rec_pca['PC1'], rec_pca['PC2'],
                c='orange', s=150, marker='*', edgecolors='black', 
                label='Önerilen Şarkılar', zorder=10
            )
        # Kullanıcı merkezini çiz
        plt.scatter(
            user_pca[0, 0], user_pca[0, 1], 
            c='red', s=400, marker='X', edgecolors='white', linewidth=3,
            label='SİZİN ZEVK MERKEZİNİZ', zorder=11
        )
        
        plt.title(f'Müzik Uzayında Konumunuz ve Öneriler (Cluster {predicted_cluster})', fontsize=16)
        plt.xlabel('Principal Component 1', fontsize=12)
        plt.ylabel('Principal Component 2', fontsize=12)
        plt.legend(loc='lower right', frameon=True, framealpha=0.9, shadow=True)
        plt.grid(True, alpha=0.2, linestyle='--')
        
        print(f"Harita oluşturuldu. Sizin bölgeniz: Cluster {predicted_cluster}")
        plt.show()

    def visualize_correlation(self):
        plt.figure(figsize=(12, 10))
        corr = self.df[self.feature_cols].corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
        plt.title('Müzik Özellikleri Korelasyon Matrisi')
        plt.show()

    def visualize_radar(self):
        if self.user_profile_vector is None or self.last_recommendations is None:
            print("Hata: Önce şarkı önerisi (recommend_songs) yapmalısınız.")
            return

        print("\nRadar grafiği hazırlanıyor...")
        
        categories = ['acousticness', 'danceability', 'energy', 
                     'instrumentalness', 'liveness', 'speechiness', 'valence','popularity', 'tempo', 'loudness']
        
        user_values = self.user_profile_vector[categories].values.flatten().tolist()
        user_values += user_values[:1] 
        # Önerilen şarkıların medyanını al
        rec_mean = self.last_recommendations[categories].median(axis=0)
        rec_values = rec_mean.values.flatten().tolist()
        rec_values += rec_values[:1] 


        angles = [n / float(len(categories)) * 2 * np.pi for n in range(len(categories))]
        angles += angles[:1]

        plt.figure(figsize=(8, 8))
        ax = plt.subplot(111, polar=True)

        plt.xticks(angles[:-1], [c.capitalize() for c in categories], color='grey', size=10)
 
        ax.plot(angles, user_values, linewidth=2, linestyle='solid', label="Sizin Tarzınız", color='blue')
        ax.fill(angles, user_values, 'blue', alpha=0.1)

        ax.plot(angles, rec_values, linewidth=2, linestyle='solid', label="Sistem Önerisi", color='red')
        ax.fill(angles, rec_values, 'red', alpha=0.1)
        
        plt.title('Girdi vs Çıktı Karşılaştırması (Eski vs Yeni)', size=15, y=1.1)
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        plt.show()