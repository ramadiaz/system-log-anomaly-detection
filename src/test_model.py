import pandas as pd
import numpy as np
from preprocessing import LogPreprocessor
from model import AnomalyDetector
from utils import generate_report
import os
import logging
from datetime import datetime

def generate_anomaly_list(logs, predictions, scores, output_path):
    """
    Menghasilkan laporan detail yang berisi daftar anomali yang terdeteksi beserta skornya dalam format CSV.
    
    Parameter:
        logs (list): Daftar entri log
        predictions (array): Array prediksi anomali (0 atau 1)
        scores (array): Array skor anomali
        output_path (str): Path untuk menyimpan laporan anomali
    """
    anomaly_indices = np.where(predictions == 1)[0]
    
    # Create a DataFrame with the anomaly information
    anomaly_data = {
        'Index': anomaly_indices + 1,  # 1-based indexing for readability
        'Log_Entry': [logs[idx] for idx in anomaly_indices],
        'Anomaly_Score': [scores[idx] for idx in anomaly_indices]
    }
    
    # Create DataFrame and sort by anomaly score in descending order
    df = pd.DataFrame(anomaly_data)
    df = df.sort_values('Anomaly_Score', ascending=False)
    
    # Add timestamp to the output
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Save to CSV with additional metadata
    with open(output_path, 'w') as f:
        f.write(f"Laporan Deteksi Anomali\n")
        f.write(f"Dibuat pada: {timestamp}\n")
        f.write(f"Total anomali terdeteksi: {len(anomaly_indices)}\n\n")
    
    # Append the DataFrame to the same file
    df.to_csv(output_path, mode='a', index=False)

def test_model():
    """
    Menguji model deteksi anomali dengan dataset yang telah disiapkan.
    Fungsi ini akan:
    1. Memuat dataset pengujian
    2. Memproses data log
    3. Membuat prediksi menggunakan model yang telah dilatih
    4. Menghasilkan laporan dan metrik evaluasi
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    print("Memuat dataset pengujian...")
    test_df = pd.read_csv('data/test_logs.csv')
    print(f"Total entri pengujian: {len(test_df)}")
    print(f"Anomali aktual dalam set pengujian: {test_df['label'].sum()}")
    
    # Calculate contamination based on actual anomaly ratio
    anomaly_ratio = test_df['label'].mean()
    print(f"Rasio anomali dalam dataset: {anomaly_ratio:.4f}")
    
    # Initialize components with calculated contamination
    preprocessor = LogPreprocessor()
    detector = AnomalyDetector(
        contamination=max(anomaly_ratio, 0.1),  # Use actual ratio or minimum 0.1
        threshold_percentile=50
    )
    
    # Load trained model and scaler
    print("Memuat model dan scaler yang telah dilatih...")
    try:
        detector.load_model('models/anomaly_detector.joblib')
        preprocessor.load_scaler('models/scaler.joblib')
    except FileNotFoundError:
        print("Error: File model tidak ditemukan. Silakan latih model terlebih dahulu menggunakan:")
        print("python src/train_model.py")
        return
    
    # Process the test data
    print("Memproses data pengujian...")
    processed_data = preprocessor.preprocess_logs(test_df['log'].tolist())
    
    # Make predictions
    print("Membuat prediksi...")
    predictions = detector.predict(processed_data)
    scores = detector.predict_proba(processed_data)
    
    # Generate reports
    print("Menghasilkan laporan...")
    report_path = os.path.join('reports', 'test_report.txt')
    generate_report(test_df['log'].tolist(), predictions, scores, report_path)
    
    # Generate anomaly list
    anomaly_list_path = os.path.join('reports', 'anomaly_list.csv')
    generate_anomaly_list(test_df['log'].tolist(), predictions, scores, anomaly_list_path)
    
    # Calculate metrics
    true_anomalies = test_df['label'].values
    predicted_anomalies = predictions
    
    # Calculate accuracy
    accuracy = np.mean(true_anomalies == predicted_anomalies)
    
    # Calculate precision and recall
    true_positives = np.sum((true_anomalies == 1) & (predicted_anomalies == 1))
    false_positives = np.sum((true_anomalies == 0) & (predicted_anomalies == 1))
    false_negatives = np.sum((true_anomalies == 1) & (predicted_anomalies == 0))
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    
    print("\nHasil Pengujian:")
    print(f"Total log yang dianalisis: {len(test_df)}")
    print(f"Anomali aktual dalam dataset: {true_anomalies.sum()}")
    print(f"Anomali yang terdeteksi: {predicted_anomalies.sum()}")
    print(f"Akurasi: {accuracy:.4f}")
    print(f"Presisi: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"\nLaporan detail disimpan di: {report_path}")
    print(f"Daftar anomali disimpan di: {anomaly_list_path}")

if __name__ == "__main__":
    test_model() 