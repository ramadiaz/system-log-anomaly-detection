import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os
import warnings


class AnomalyDetector:
    def __init__(self, contamination='auto', threshold_percentile=95, validation_split=0.2):
        """
        Inisialisasi detektor anomali dengan konfigurasi anti-overfitting.

        Parameter:
            contamination (str/float): Proporsi outlier ('auto' untuk estimasi otomatis)
            threshold_percentile (float): Persentil untuk threshold (95 lebih konservatif)
            validation_split (float): Proporsi data untuk validasi
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
        self.validation_split = validation_split
        self.threshold = None
        self.scaler = StandardScaler()  # Tambahkan normalisasi
        self.train_scores = None
        self.val_scores = None

    def _validate_data(self, X):
        """Validasi dan preprocessing data"""
        if X is None or len(X) == 0:
            raise ValueError("Data tidak boleh kosong")

        # Konversi ke numpy array jika belum
        X = np.array(X)

        # Check untuk nilai NaN atau infinite
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            warnings.warn("Data mengandung NaN atau infinite values. Akan dibersihkan otomatis.")
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        return X

    def train(self, X):
        """
        Melatih model dengan validasi untuk mencegah overfitting.

        Parameter:
            X (numpy.ndarray): Data pelatihan
        """
        X = self._validate_data(X)

        # Normalisasi data
        X_scaled = self.scaler.fit_transform(X)

        # Split data untuk training dan validasi
        if len(X_scaled) > 100:  # Hanya split jika data cukup besar
            X_train, X_val = train_test_split(
                X_scaled,
                test_size=self.validation_split,
                random_state=42,
                shuffle=True
            )
        else:
            X_train = X_scaled
            X_val = X_scaled

        # Latih model pada training set
        self.model.fit(X_train)

        # Hitung scores pada training dan validation set
        self.train_scores = -self.model.score_samples(X_train)
        self.val_scores = -self.model.score_samples(X_val)

        # Gunakan validation scores untuk menentukan threshold (lebih robust)
        self.threshold = np.percentile(self.val_scores, self.threshold_percentile)

        # Statistik untuk monitoring
        train_anomalies = np.sum(self.train_scores > self.threshold)
        val_anomalies = np.sum(self.val_scores > self.threshold)

        print(f"=== Training Summary ===")
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Threshold: {self.threshold:.4f}")
        print(f"Training anomalies: {train_anomalies} ({train_anomalies / len(X_train) * 100:.2f}%)")
        print(f"Validation anomalies: {val_anomalies} ({val_anomalies / len(X_val) * 100:.2f}%)")

        # Warning jika ada indikasi overfitting
        if len(X_train) != len(X_val):
            train_anomaly_rate = train_anomalies / len(X_train)
            val_anomaly_rate = val_anomalies / len(X_val)

            if abs(train_anomaly_rate - val_anomaly_rate) > 0.05:  # 5% difference
                warnings.warn(
                    f"Possible overfitting detected! "
                    f"Training anomaly rate: {train_anomaly_rate:.3f}, "
                    f"Validation anomaly rate: {val_anomaly_rate:.3f}"
                )

    def predict(self, X):
        """
        Memprediksi anomali dengan preprocessing yang konsisten.

        Parameter:
            X (numpy.ndarray): Data yang akan diprediksi

        Returns:
            numpy.ndarray: 1 untuk anomali, 0 untuk normal
        """
        if self.threshold is None:
            raise ValueError("Model belum dilatih. Panggil train() terlebih dahulu.")

        X = self._validate_data(X)
        X_scaled = self.scaler.transform(X)  # Gunakan scaler yang sudah difit

        scores = -self.model.score_samples(X_scaled)
        predictions = (scores > self.threshold).astype(int)

        return predictions

    def predict_proba(self, X):
        """
        Mendapatkan skor anomali yang sudah dinormalisasi.

        Parameter:
            X (numpy.ndarray): Data yang akan diprediksi

        Returns:
            numpy.ndarray: Skor anomali (0-1, nilai tinggi = lebih anomali)
        """
        if self.threshold is None:
            raise ValueError("Model belum dilatih. Panggil train() terlebih dahulu.")

        X = self._validate_data(X)
        X_scaled = self.scaler.transform(X)

        raw_scores = -self.model.score_samples(X_scaled)

        # Normalisasi scores ke range 0-1 berdasarkan training data
        if self.val_scores is not None:
            min_score = np.min(self.val_scores)
            max_score = np.max(self.val_scores)
            normalized_scores = (raw_scores - min_score) / (max_score - min_score)
            # Clip ke range 0-1
            normalized_scores = np.clip(normalized_scores, 0, 1)
            return normalized_scores
        else:
            return raw_scores

    def get_feature_importance(self):
        """
        Estimasi pentingnya features (experimental).

        Returns:
            numpy.ndarray: Skor pentingnya setiap feature
        """
        if hasattr(self.model, 'estimators_'):
            # Hitung rata-rata path length untuk setiap feature
            importances = np.zeros(self.model.n_features_in_)

            for estimator in self.model.estimators_:
                if hasattr(estimator, 'tree_'):
                    tree = estimator.tree_
                    feature_importances = tree.compute_feature_importances()
                    importances += feature_importances

            importances /= len(self.model.estimators_)
            return importances
        else:
            warnings.warn("Feature importance tidak tersedia sebelum training")
            return None

    def save_model(self, path):
        """
        Menyimpan model dan semua komponennya.

        Parameter:
            path (str): Path untuk menyimpan model
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'threshold': self.threshold,
            'threshold_percentile': self.threshold_percentile,
            'validation_split': self.validation_split,
            'train_scores': self.train_scores,
            'val_scores': self.val_scores
        }
        joblib.dump(model_data, path)
        print(f"Model berhasil disimpan ke: {path}")

    def load_model(self, path):
        """
        Memuat model yang tersimpan.

        Parameter:
            path (str): Path ke model yang tersimpan
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"File model tidak ditemukan: {path}")

        model_data = joblib.load(path)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.threshold = model_data['threshold']
        self.threshold_percentile = model_data.get('threshold_percentile', 95)
        self.validation_split = model_data.get('validation_split', 0.2)
        self.train_scores = model_data.get('train_scores')
        self.val_scores = model_data.get('val_scores')
        print(f"Model berhasil dimuat dari: {path}")


# Contoh penggunaan yang lebih robust
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    normal_data = np.random.normal(0, 1, (1000, 5))
    anomaly_data = np.random.normal(3, 1, (50, 5))
    data = np.vstack([normal_data, anomaly_data])

    # Inisialisasi detector dengan konfigurasi anti-overfitting
    detector = AnomalyDetector(
        contamination='auto',  # Estimasi otomatis
        threshold_percentile=95,  # Lebih konservatif
        validation_split=0.2
    )

    # Latih model
    detector.train(data)

    # Test prediksi
    test_data = np.random.normal(0, 1, (100, 5))
    predictions = detector.predict(test_data)
    probabilities = detector.predict_proba(test_data)

    print(f"\nHasil prediksi:")
    print(f"Anomali terdeteksi: {np.sum(predictions)}/{len(predictions)}")
    print(f"Rata-rata skor anomali: {np.mean(probabilities):.4f}")