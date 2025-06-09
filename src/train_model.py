import pandas as pd
from preprocessing import LogPreprocessor
from model import AnomalyDetector
import os
import logging
import time
import numpy as np
from tqdm import tqdm
import warnings


def calculate_optimal_contamination(labels, min_contamination=0.01, max_contamination=0.3):
    """
    Hitung kontaminasi optimal berdasarkan label aktual dengan batasan yang masuk akal.

    Parameter:
        labels: Array label (1 untuk anomali, 0 untuk normal)
        min_contamination: Kontaminasi minimum
        max_contamination: Kontaminasi maksimum

    Returns:
        float: Kontaminasi optimal
    """
    if labels is not None and len(labels) > 0:
        actual_ratio = np.mean(labels)
        # Batasi kontaminasi dalam range yang masuk akal
        optimal_contamination = np.clip(actual_ratio, min_contamination, max_contamination)
        return optimal_contamination
    else:
        return 'auto'  # Gunakan estimasi otomatis jika tidak ada label


def validate_dataset(df):
    """
    Validasi dataset sebelum training.

    Parameter:
        df: DataFrame dengan kolom 'log' dan 'label'

    Returns:
        bool: True jika dataset valid
    """
    required_columns = ['log', 'label']
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Kolom yang diperlukan tidak ditemukan: {missing_columns}")

    # Check untuk data kosong
    if df.empty:
        raise ValueError("Dataset kosong")

    # Check untuk nilai null
    null_logs = df['log'].isnull().sum()
    null_labels = df['label'].isnull().sum()

    if null_logs > 0:
        logging.warning(f"Ditemukan {null_logs} log entry null. Akan dihapus.")

    if null_labels > 0:
        logging.warning(f"Ditemukan {null_labels} label null. Akan dihapus.")

    # Bersihkan data null
    df_clean = df.dropna(subset=['log', 'label'])

    if len(df_clean) < len(df) * 0.5:  # Jika lebih dari 50% data hilang
        raise ValueError("Terlalu banyak data null. Dataset tidak valid untuk training.")

    return df_clean


def train_model():
    """
    Melatih model deteksi anomali dengan konfigurasi anti-overfitting.
    Fungsi ini akan:
    1. Memuat dan validasi dataset pelatihan
    2. Menghitung kontaminasi optimal
    3. Memproses data log dengan preprocessing yang robust
    4. Melatih model dengan validasi split
    5. Menyimpan model dan preprocessor
    """
    # Setup logging yang lebih detail
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    start_time = time.time()

    try:
        # 1. Load dan validasi dataset
        logging.info("=" * 50)
        logging.info("MEMULAI TRAINING MODEL ANOMALY DETECTION")
        logging.info("=" * 50)

        logging.info("Memuat dataset pelatihan...")
        if not os.path.exists('data/train_logs.csv'):
            raise FileNotFoundError("File 'data/train_logs.csv' tidak ditemukan")

        train_df = pd.read_csv('data/train_logs.csv')
        logging.info(f"Dataset awal dimuat: {len(train_df)} entri")

        # Validasi dan bersihkan dataset
        train_df = validate_dataset(train_df)
        logging.info(f"Dataset setelah validasi: {len(train_df)} entri")

        # 2. Analisis dataset
        logging.info("-" * 30)
        logging.info("ANALISIS DATASET")
        logging.info("-" * 30)

        total_logs = len(train_df)
        anomaly_count = train_df['label'].sum()
        normal_count = total_logs - anomaly_count
        anomaly_ratio = anomaly_count / total_logs

        logging.info(f"Total logs: {total_logs:,}")
        logging.info(f"Normal logs: {normal_count:,} ({(1 - anomaly_ratio) * 100:.2f}%)")
        logging.info(f"Anomaly logs: {anomaly_count:,} ({anomaly_ratio * 100:.2f}%)")

        # Peringatan jika dataset tidak seimbang
        if anomaly_ratio < 0.01:
            logging.warning("Dataset sangat tidak seimbang (< 1% anomali). Model mungkin bias.")
        elif anomaly_ratio > 0.4:
            logging.warning("Terlalu banyak anomali (> 40%). Periksa kembali labeling data.")

        # 3. Hitung kontaminasi optimal
        optimal_contamination = calculate_optimal_contamination(train_df['label'].values)
        logging.info(f"Kontaminasi optimal: {optimal_contamination}")

        # 4. Inisialisasi komponen dengan konfigurasi anti-overfitting
        logging.info("-" * 30)
        logging.info("INISIALISASI KOMPONEN")
        logging.info("-" * 30)

        preprocessor = LogPreprocessor()

        # Konfigurasi detector dengan parameter anti-overfitting
        detector = AnomalyDetector(
            contamination=optimal_contamination,
            threshold_percentile=95,  # Lebih konservatif untuk mengurangi false positive
            validation_split=0.2  # 20% data untuk validasi
        )

        logging.info("Preprocessor dan Detector berhasil diinisialisasi")
        logging.info(f"Konfigurasi Detector:")
        logging.info(f"  - Contamination: {detector.model.contamination}")
        logging.info(f"  - N Estimators: {detector.model.n_estimators}")
        logging.info(f"  - Max Samples: {detector.model.max_samples}")
        logging.info(f"  - Max Features: {detector.model.max_features}")
        logging.info(f"  - Threshold Percentile: {detector.threshold_percentile}")
        logging.info(f"  - Validation Split: {detector.validation_split}")

        # 5. Preprocessing data
        logging.info("-" * 30)
        logging.info("PREPROCESSING DATA")
        logging.info("-" * 30)

        logging.info("Memproses data pelatihan (ekstraksi fitur dan normalisasi)...")
        t0 = time.time()

        # Extract logs untuk preprocessing
        log_texts = train_df['log'].tolist()
        processed_data = preprocessor.preprocess_logs(log_texts, progress_bar=True)

        preprocessing_time = time.time() - t0
        logging.info(f"Bentuk data yang diproses: {processed_data.shape}")
        logging.info(f"Pemrosesan selesai dalam {preprocessing_time:.2f} detik")

        # Validasi hasil preprocessing
        if processed_data.shape[0] != len(train_df):
            raise ValueError("Ukuran data hasil preprocessing tidak sesuai dengan dataset asli")

        # Check untuk nilai infinite atau NaN
        if np.any(np.isnan(processed_data)) or np.any(np.isinf(processed_data)):
            logging.warning("Data mengandung NaN atau infinite values. Akan dibersihkan otomatis.")

        # 6. Training model dengan validasi
        logging.info("-" * 30)
        logging.info("TRAINING MODEL")
        logging.info("-" * 30)

        logging.info("Melatih model dengan validasi split...")
        t0 = time.time()

        # Training dengan monitoring
        detector.train(processed_data)

        training_time = time.time() - t0
        logging.info(f"Training selesai dalam {training_time:.2f} detik")

        # 7. Evaluasi pada training data untuk monitoring
        logging.info("-" * 30)
        logging.info("EVALUASI MODEL")
        logging.info("-" * 30)

        # Prediksi pada training data
        train_predictions = detector.predict(processed_data)
        train_scores = detector.predict_proba(processed_data)

        # Hitung metrik
        true_positives = np.sum((train_df['label'] == 1) & (train_predictions == 1))
        false_positives = np.sum((train_df['label'] == 0) & (train_predictions == 1))
        true_negatives = np.sum((train_df['label'] == 0) & (train_predictions == 0))
        false_negatives = np.sum((train_df['label'] == 1) & (train_predictions == 0))

        # Hitung precision, recall, f1-score
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        logging.info("Hasil evaluasi pada training data:")
        logging.info(f"  True Positives: {true_positives}")
        logging.info(f"  False Positives: {false_positives}")
        logging.info(f"  True Negatives: {true_negatives}")
        logging.info(f"  False Negatives: {false_negatives}")
        logging.info(f"  Precision: {precision:.4f}")
        logging.info(f"  Recall: {recall:.4f}")
        logging.info(f"  F1-Score: {f1_score:.4f}")
        logging.info(f"  Rata-rata anomaly score: {np.mean(train_scores):.4f}")

        # 8. Simpan model dan komponen
        logging.info("-" * 30)
        logging.info("MENYIMPAN MODEL")
        logging.info("-" * 30)

        os.makedirs('models', exist_ok=True)

        # Simpan detector (sudah include scaler internal)
        detector.save_model('models/anomaly_detector.joblib')

        # Simpan preprocessor secara terpisah untuk fleksibilitas
        preprocessor.save_scaler('models/scaler.joblib')

        # Simpan metadata training
        training_metadata = {
            'dataset_size': len(train_df),
            'anomaly_ratio': anomaly_ratio,
            'contamination_used': optimal_contamination,
            'preprocessing_time': preprocessing_time,
            'training_time': training_time,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'feature_shape': processed_data.shape,
            'threshold_percentile': detector.threshold_percentile,
            'validation_split': detector.validation_split
        }

        import json
        with open('models/training_metadata.json', 'w') as f:
            json.dump(training_metadata, f, indent=2)

        # 9. Summary
        total_time = time.time() - start_time
        logging.info("=" * 50)
        logging.info("TRAINING BERHASIL DISELESAIKAN")
        logging.info("=" * 50)
        logging.info("File yang disimpan:")
        logging.info(f"  - Model: models/anomaly_detector.joblib")
        logging.info(f"  - Preprocessor: models/scaler.joblib")
        logging.info(f"  - Metadata: models/training_metadata.json")
        logging.info(f"Total waktu training: {total_time:.2f} detik")
        logging.info(f"Model siap digunakan untuk deteksi anomali!")

        return detector, preprocessor, training_metadata

    except Exception as e:
        logging.error(f"Error during training: {str(e)}")
        logging.error("Training gagal. Periksa log error di atas.")
        raise


def load_and_test_model(test_logs=None):
    """
    Fungsi tambahan untuk test model yang sudah dilatih.

    Parameter:
        test_logs (list): Daftar log untuk ditest (opsional)
    """
    try:
        logging.info("Memuat model yang sudah dilatih...")

        # Load components
        detector = AnomalyDetector()
        detector.load_model('models/anomaly_detector.joblib')

        preprocessor = LogPreprocessor()
        preprocessor.load_scaler('models/scaler.joblib')

        logging.info("Model berhasil dimuat!")

        if test_logs:
            logging.info(f"Testing pada {len(test_logs)} log entries...")
            processed_test = preprocessor.preprocess_logs(test_logs, progress_bar=True)
            predictions = detector.predict(processed_test)
            scores = detector.predict_proba(processed_test)

            anomaly_count = np.sum(predictions)
            logging.info(f"Hasil test: {anomaly_count}/{len(test_logs)} anomali terdeteksi")
            logging.info(f"Rata-rata anomaly score: {np.mean(scores):.4f}")

            return predictions, scores

    except Exception as e:
        logging.error(f"Error saat testing model: {str(e)}")
        raise


if __name__ == "__main__":
    try:
        # Training
        detector, preprocessor, metadata = train_model()

        # Optional: Test dengan beberapa sample log
        sample_test_logs = [
            "INFO: User login successful for user123",
            "ERROR: Database connection failed after 3 retries",
            "WARNING: High memory usage detected (95%)",
        ]

        logging.info("\nTesting model dengan sample logs...")
        predictions, scores = load_and_test_model(sample_test_logs)

        for i, (log, pred, score) in enumerate(zip(sample_test_logs, predictions, scores)):
            status = "ANOMALI" if pred == 1 else "NORMAL"
            logging.info(f"Test {i + 1}: {status} (score: {score:.4f}) - {log}")

    except Exception as e:
        logging.error(f"Script gagal: {str(e)}")
        exit(1)