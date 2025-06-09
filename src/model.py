import numpy as np
from sklearn.ensemble import IsolationForest
import joblib
import os

class AnomalyDetector:
    def __init__(self, contamination=0.5, threshold_percentile=50):
        """
        Inisialisasi detektor anomali.
        
        Parameter:
            contamination (float): Proporsi outlier dalam dataset.
            threshold_percentile (float): Persentil yang digunakan sebagai ambang batas untuk skor anomali.
        """
        self.model = IsolationForest(
            n_estimators=50,  # Dikurangi dari 100 untuk mengurangi overfitting
            max_samples=0.8,  # Gunakan 80% sample per tree (lebih robust)
            contamination=contamination,  # 'auto' lebih adaptif
            max_features=0.8,  # Gunakan 80% features per tree
            random_state=42,
            n_jobs=-1,  # Gunakan semua core yang tersedia
            verbose=0,  # Kurangi noise output
            bootstrap=True  # Enable bootstrap sampling
        )
        self.threshold_percentile = threshold_percentile
        self.threshold = None
        
    def train(self, X):
        """
        Melatih model deteksi anomali.
        
        Parameter:
            X (numpy.ndarray): Data pelatihan
        """
        self.model.fit(X)
        
        scores = -self.model.score_samples(X)
        
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        percentile_threshold = np.percentile(scores, self.threshold_percentile)
        
        self.threshold = min(percentile_threshold, mean_score + std_score)
        
        print(f"Ambang batas anomali ditetapkan ke: {self.threshold:.4f}")
        print(f"Jumlah anomali dalam set pelatihan: {np.sum(scores > self.threshold)}")
        
    def predict(self, X):
        """
        Memprediksi anomali dalam data.
        
        Parameter:
            X (numpy.ndarray): Data yang akan diprediksi
            
        Returns:
            numpy.ndarray: 1 untuk anomali, 0 untuk normal
        """
        scores = -self.model.score_samples(X)
        return (scores > self.threshold).astype(int)
    
    def predict_proba(self, X):
        """
        Mendapatkan skor anomali untuk data.
        
        Parameter:
            X (numpy.ndarray): Data yang akan diprediksi
            
        Returns:
            numpy.ndarray: Skor anomali (nilai lebih tinggi berarti lebih anomali)
        """
        return -self.model.score_samples(X)
    
    def save_model(self, path):
        """
        Menyimpan model ke disk.
        
        Parameter:
            path (str): Path untuk menyimpan model
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        model_data = {
            'model': self.model,
            'threshold': self.threshold,
            'threshold_percentile': self.threshold_percentile
        }
        joblib.dump(model_data, path)
    
    def load_model(self, path):
        """
        Memuat model yang tersimpan dari disk.
        
        Parameter:
            path (str): Path ke model yang tersimpan
        """
        model_data = joblib.load(path)
        self.model = model_data['model']
        self.threshold = model_data['threshold']
        self.threshold_percentile = model_data['threshold_percentile'] 