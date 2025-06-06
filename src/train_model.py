import pandas as pd
from preprocessing import LogPreprocessor
from model import AnomalyDetector
import os
import logging
import time
from tqdm import tqdm

def train_model():
    """
    Melatih model deteksi anomali dengan dataset yang telah disiapkan.
    Fungsi ini akan:
    1. Memuat dataset pelatihan
    2. Memproses data log
    3. Melatih model deteksi anomali
    4. Menyimpan model dan scaler yang telah dilatih
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    start_time = time.time()
    logging.info("Memuat dataset pelatihan...")
    test_df = pd.read_csv('data/train_logs.csv')
    logging.info(f"Memuat {len(test_df)} entri log.")
    
    # Hitung kontaminasi berdasarkan rasio anomali aktual
    anomaly_ratio = test_df['label'].mean()
    logging.info(f"Rasio anomali dalam dataset: {anomaly_ratio:.4f}")
    
    # Inisialisasi komponen dengan kontaminasi yang dihitung
    preprocessor = LogPreprocessor()
    detector = AnomalyDetector(
        contamination=max(anomaly_ratio, 0.1),  # Gunakan rasio aktual atau minimum 0.1
        threshold_percentile=50
    )
    
    # Proses data pelatihan
    logging.info("Memproses data pelatihan (ekstraksi fitur dan penskalaan)...")
    t0 = time.time()
    processed_data = preprocessor.preprocess_logs(test_df['log'].tolist(), progress_bar=True)
    logging.info(f"Bentuk data yang diproses: {processed_data.shape}")
    logging.info(f"Pemrosesan selesai dalam {time.time() - t0:.2f} detik.")
    
    # Latih model
    logging.info("Melatih model...")
    t0 = time.time()
    detector.train(processed_data)
    logging.info(f"Pelatihan model selesai dalam {time.time() - t0:.2f} detik.")
    
    # Simpan model dan scaler
    logging.info("Menyimpan model dan scaler...")
    os.makedirs('models', exist_ok=True)
    detector.save_model('models/anomaly_detector.joblib')
    preprocessor.save_scaler('models/scaler.joblib')
    logging.info("Model dan scaler berhasil disimpan.")
    logging.info(f"Model disimpan ke: models/anomaly_detector.joblib")
    logging.info(f"Scaler disimpan ke: models/scaler.joblib")
    logging.info(f"Total waktu pelatihan: {time.time() - start_time:.2f} detik.")

if __name__ == "__main__":
    train_model() 