import pandas as pd
import numpy as np
from preprocessing import LogPreprocessor
from model import AnomalyDetector
from utils import generate_report
import os
import logging
import json
import time
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def validate_test_dataset(df):
    """
    Validasi dataset test sebelum evaluasi.

    Parameter:
        df: DataFrame dengan kolom 'log' dan 'label'

    Returns:
        DataFrame: Dataset yang sudah dibersihkan
    """
    required_columns = ['log', 'label']
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Kolom yang diperlukan tidak ditemukan: {missing_columns}")

    if df.empty:
        raise ValueError("Dataset test kosong")

    # Bersihkan data null
    initial_size = len(df)
    df_clean = df.dropna(subset=['log', 'label'])

    if len(df_clean) < initial_size:
        logging.warning(f"Menghapus {initial_size - len(df_clean)} entries dengan nilai null")

    if len(df_clean) == 0:
        raise ValueError("Tidak ada data valid setelah pembersihan")

    return df_clean


def generate_anomaly_list(logs, predictions, scores, output_path, metadata=None):
    """
    Menghasilkan laporan detail yang berisi daftar anomali yang terdeteksi beserta skornya dalam format CSV.

    Parameter:
        logs (list): Daftar entri log
        predictions (array): Array prediksi anomali (0 atau 1)
        scores (array): Array skor anomali (0-1 normalized)
        output_path (str): Path untuk menyimpan laporan anomali
        metadata (dict): Metadata tambahan untuk laporan
    """
    anomaly_indices = np.where(predictions == 1)[0]

    if len(anomaly_indices) == 0:
        logging.warning("Tidak ada anomali yang terdeteksi!")

    # Create a DataFrame with the anomaly information
    anomaly_data = {
        'Index': anomaly_indices + 1,  # 1-based indexing for readability
        'Log_Entry': [logs[idx] for idx in anomaly_indices],
        'Anomaly_Score': [scores[idx] for idx in anomaly_indices],
        'Confidence': ['High' if scores[idx] > 0.8 else 'Medium' if scores[idx] > 0.6 else 'Low'
                       for idx in anomaly_indices]
    }

    # Create DataFrame and sort by anomaly score in descending order
    df = pd.DataFrame(anomaly_data)
    df = df.sort_values('Anomaly_Score', ascending=False)

    # Add timestamp and metadata
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save to CSV with additional metadata
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"# Laporan Deteksi Anomali\n")
        f.write(f"# Dibuat pada: {timestamp}\n")
        f.write(f"# Total log dianalisis: {len(logs)}\n")
        f.write(f"# Total anomali terdeteksi: {len(anomaly_indices)}\n")
        f.write(f"# Persentase anomali: {len(anomaly_indices) / len(logs) * 100:.2f}%\n")

        if metadata:
            f.write(f"# Model threshold: {metadata.get('threshold', 'N/A')}\n")
            f.write(f"# Threshold percentile: {metadata.get('threshold_percentile', 'N/A')}\n")

        f.write(f"#\n")

    # Append the DataFrame to the same file
    df.to_csv(output_path, mode='a', index=False)
    logging.info(f"Anomaly list saved to: {output_path}")


def generate_detailed_metrics(true_labels, predictions, scores, output_dir):
    """
    Menghasilkan metrik evaluasi yang detail dan visualisasi.

    Parameter:
        true_labels: Label sebenarnya
        predictions: Prediksi model
        scores: Skor anomali
        output_dir: Directory untuk menyimpan hasil
    """
    os.makedirs(output_dir, exist_ok=True)

    # Calculate detailed metrics
    tn, fp, fn, tp = confusion_matrix(true_labels, predictions).ravel()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    # Additional metrics
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0

    metrics = {
        'confusion_matrix': {
            'true_positive': int(tp),
            'true_negative': int(tn),
            'false_positive': int(fp),
            'false_negative': int(fn)
        },
        'performance_metrics': {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1_score),
            'specificity': float(specificity),
            'false_positive_rate': float(false_positive_rate),
            'false_negative_rate': float(false_negative_rate)
        },
        'score_statistics': {
            'mean_score': float(np.mean(scores)),
            'std_score': float(np.std(scores)),
            'min_score': float(np.min(scores)),
            'max_score': float(np.max(scores)),
            'median_score': float(np.median(scores))
        },
        'dataset_info': {
            'total_samples': len(true_labels),
            'actual_anomalies': int(np.sum(true_labels)),
            'predicted_anomalies': int(np.sum(predictions)),
            'actual_anomaly_rate': float(np.mean(true_labels)),
            'predicted_anomaly_rate': float(np.mean(predictions))
        }
    }

    # Save detailed metrics to JSON
    with open(os.path.join(output_dir, 'detailed_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    return metrics


def load_training_metadata():
    """
    Load metadata dari training untuk perbandingan.

    Returns:
        dict: Training metadata atau None jika tidak ada
    """
    metadata_path = 'models/training_metadata.json'
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logging.warning(f"Gagal memuat training metadata: {e}")
    return None


def test_model():
    """
    Menguji model deteksi anomali dengan dataset yang telah disiapkan.
    Fungsi ini akan:
    1. Memuat dan validasi dataset pengujian
    2. Memuat model yang sudah trained
    3. Memproses data log dengan preprocessing yang konsisten
    4. Membuat prediksi menggunakan model
    5. Menghasilkan laporan dan metrik evaluasi yang detail
    6. Membandingkan dengan hasil training
    """
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    start_time = time.time()

    try:
        # 1. Load dan validasi dataset test
        logging.info("=" * 50)
        logging.info("MEMULAI EVALUASI MODEL ANOMALY DETECTION")
        logging.info("=" * 50)

        logging.info("Memuat dataset pengujian...")
        if not os.path.exists('data/test_logs.csv'):
            raise FileNotFoundError("File 'data/test_logs.csv' tidak ditemukan")

        test_df = pd.read_csv('data/test_logs.csv')
        logging.info(f"Dataset test awal: {len(test_df)} entries")

        # Validasi dataset
        test_df = validate_test_dataset(test_df)
        logging.info(f"Dataset test setelah validasi: {len(test_df)} entries")

        # Analisis dataset test
        total_test = len(test_df)
        actual_anomalies = test_df['label'].sum()
        actual_anomaly_rate = actual_anomalies / total_test

        logging.info(f"Total test logs: {total_test:,}")
        logging.info(f"Actual anomalies: {actual_anomalies:,} ({actual_anomaly_rate * 100:.2f}%)")

        # 2. Load model dan komponen
        logging.info("-" * 30)
        logging.info("MEMUAT MODEL DAN PREPROCESSOR")
        logging.info("-" * 30)

        # Initialize components
        preprocessor = LogPreprocessor()
        detector = AnomalyDetector()

        # Load trained model dan scaler
        try:
            detector.load_model('models/anomaly_detector.joblib')
            preprocessor.load_scaler('models/scaler.joblib')
            logging.info("Model dan preprocessor berhasil dimuat")
        except FileNotFoundError as e:
            logging.error(f"File model tidak ditemukan: {e}")
            logging.error("Silakan latih model terlebih dahulu dengan menjalankan:")
            logging.error("python src/train_model.py")
            return

        # Load training metadata untuk perbandingan
        training_metadata = load_training_metadata()
        if training_metadata:
            logging.info("Training metadata dimuat untuk perbandingan")
            logging.info(f"Training anomaly rate: {training_metadata.get('anomaly_ratio', 'N/A'):.4f}")
            logging.info(f"Training F1-score: {training_metadata.get('f1_score', 'N/A'):.4f}")

        # 3. Preprocessing data test
        logging.info("-" * 30)
        logging.info("PREPROCESSING DATA TEST")
        logging.info("-" * 30)

        logging.info("Memproses data test dengan preprocessing yang konsisten...")
        t0 = time.time()

        log_texts = test_df['log'].tolist()
        processed_data = preprocessor.preprocess_logs(log_texts, progress_bar=True)

        preprocessing_time = time.time() - t0
        logging.info(f"Preprocessing selesai dalam {preprocessing_time:.2f} detik")
        logging.info(f"Shape data yang diproses: {processed_data.shape}")

        # 4. Prediksi
        logging.info("-" * 30)
        logging.info("MEMBUAT PREDIKSI")
        logging.info("-" * 30)

        logging.info("Membuat prediksi anomali...")
        t0 = time.time()

        predictions = detector.predict(processed_data)
        scores = detector.predict_proba(processed_data)  # Normalized scores 0-1

        prediction_time = time.time() - t0
        logging.info(f"Prediksi selesai dalam {prediction_time:.2f} detik")

        predicted_anomalies = np.sum(predictions)
        predicted_anomaly_rate = predicted_anomalies / total_test

        logging.info(f"Anomali terdeteksi: {predicted_anomalies:,} ({predicted_anomaly_rate * 100:.2f}%)")

        # 5. Evaluasi dan metrik
        logging.info("-" * 30)
        logging.info("EVALUASI PERFORMA MODEL")
        logging.info("-" * 30)

        # Generate detailed metrics
        os.makedirs('reports', exist_ok=True)
        detailed_metrics = generate_detailed_metrics(
            test_df['label'].values,
            predictions,
            scores,
            'reports'
        )

        # Print hasil evaluasi
        cm = detailed_metrics['confusion_matrix']
        perf = detailed_metrics['performance_metrics']

        logging.info("Confusion Matrix:")
        logging.info(f"  True Positives:  {cm['true_positive']:4d}")
        logging.info(f"  False Positives: {cm['false_positive']:4d}")
        logging.info(f"  True Negatives:  {cm['true_negative']:4d}")
        logging.info(f"  False Negatives: {cm['false_negative']:4d}")

        logging.info("Performance Metrics:")
        logging.info(f"  Accuracy:    {perf['accuracy']:.4f}")
        logging.info(f"  Precision:   {perf['precision']:.4f}")
        logging.info(f"  Recall:      {perf['recall']:.4f}")
        logging.info(f"  F1-Score:    {perf['f1_score']:.4f}")
        logging.info(f"  Specificity: {perf['specificity']:.4f}")

        # 6. Generate reports
        logging.info("-" * 30)
        logging.info("GENERATING REPORTS")
        logging.info("-" * 30)

        # Generate standard report
        report_path = os.path.join('reports', 'test_report.txt')
        generate_report(log_texts, predictions, scores, report_path)

        # Generate detailed anomaly list
        anomaly_list_path = os.path.join('reports', 'anomaly_list.csv')
        model_metadata = {
            'threshold': detector.threshold,
            'threshold_percentile': detector.threshold_percentile
        }
        generate_anomaly_list(log_texts, predictions, scores, anomaly_list_path, model_metadata)

        # 7. Perbandingan dengan training (jika tersedia)
        if training_metadata:
            logging.info("-" * 30)
            logging.info("PERBANDINGAN DENGAN TRAINING")
            logging.info("-" * 30)

            train_f1 = training_metadata.get('f1_score', 0)
            test_f1 = perf['f1_score']

            logging.info(f"Training F1-Score: {train_f1:.4f}")
            logging.info(f"Test F1-Score:     {test_f1:.4f}")
            logging.info(f"F1-Score Delta:    {test_f1 - train_f1:.4f}")

            if abs(test_f1 - train_f1) > 0.1:  # 10% difference
                if test_f1 < train_f1:
                    logging.warning("Model performance dropped significantly on test data!")
                    logging.warning("Possible overfitting or dataset mismatch.")
                else:
                    logging.info("Model performs better on test data than training data.")

        # 8. Summary
        total_time = time.time() - start_time
        logging.info("=" * 50)
        logging.info("EVALUASI SELESAI")
        logging.info("=" * 50)

        logging.info("File output yang dihasilkan:")
        logging.info(f"  - Test report: {report_path}")
        logging.info(f"  - Anomaly list: {anomaly_list_path}")
        logging.info(f"  - Detailed metrics: reports/detailed_metrics.json")

        logging.info(f"Total waktu evaluasi: {total_time:.2f} detik")
        logging.info(f"Performance summary:")
        logging.info(f"  - Accuracy: {perf['accuracy']:.4f}")
        logging.info(f"  - F1-Score: {perf['f1_score']:.4f}")
        logging.info(f"  - Anomaly detection rate: {predicted_anomaly_rate:.4f}")

        return {
            'predictions': predictions,
            'scores': scores,
            'metrics': detailed_metrics,
            'processing_times': {
                'preprocessing': preprocessing_time,
                'prediction': prediction_time,
                'total': total_time
            }
        }

    except Exception as e:
        logging.error(f"Error during model testing: {str(e)}")
        logging.error("Testing gagal. Periksa log error di atas.")
        raise


def quick_test(log_entries):
    """
    Fungsi untuk test cepat dengan beberapa log entries.

    Parameter:
        log_entries (list): Daftar log untuk ditest

    Returns:
        tuple: (predictions, scores)
    """
    try:
        logging.info(f"Quick test dengan {len(log_entries)} log entries...")

        # Load components
        preprocessor = LogPreprocessor()
        detector = AnomalyDetector()

        detector.load_model('models/anomaly_detector.joblib')
        preprocessor.load_scaler('models/scaler.joblib')

        # Process and predict
        processed_data = preprocessor.preprocess_logs(log_entries, progress_bar=False)
        predictions = detector.predict(processed_data)
        scores = detector.predict_proba(processed_data)

        # Print results
        for i, (log, pred, score) in enumerate(zip(log_entries, predictions, scores)):
            status = "ANOMALI" if pred == 1 else "NORMAL"
            confidence = "HIGH" if score > 0.8 else "MEDIUM" if score > 0.6 else "LOW"
            logging.info(f"Log {i + 1}: {status} ({confidence}, score: {score:.4f})")
            logging.info(f"  Content: {log[:100]}{'...' if len(log) > 100 else ''}")

        return predictions, scores

    except Exception as e:
        logging.error(f"Error during quick test: {str(e)}")
        raise


if __name__ == "__main__":
    try:
        # Full model testing
        results = test_model()

        # Optional: Quick test dengan sample logs
        sample_logs = [
            "INFO: User login successful for user123 from IP 192.168.1.100",
            "ERROR: Database connection failed after 3 retries - timeout exceeded",
            "WARNING: High memory usage detected (95%) - system may become unstable",
            "DEBUG: Processing batch job #12345 completed successfully",
            "CRITICAL: Security breach detected - unauthorized access attempt from IP 10.0.0.1"
        ]

        logging.info("\n" + "=" * 50)
        logging.info("QUICK TEST DENGAN SAMPLE LOGS")
        logging.info("=" * 50)

        quick_predictions, quick_scores = quick_test(sample_logs)

    except Exception as e:
        logging.error(f"Script gagal: {str(e)}")
        exit(1)