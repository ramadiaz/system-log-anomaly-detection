import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import kaggle
import zipfile
import io

def download_hdfs_dataset():
    """
    Mengunduh dataset log dari Kaggle menggunakan Kaggle API.
    Catatan: Anda perlu memiliki file kaggle.json di direktori ~/.kaggle/
    """
    # Buat direktori data jika belum ada
    os.makedirs('data', exist_ok=True)
    
    # Periksa apakah file target sudah ada
    target_file_path = os.path.join('data', 'hdfs_log', 'hdfs.log', 'sorted.log')
    if not os.path.exists(target_file_path):
        print("Mengunduh dataset dari Kaggle...")
        try:
            # Unduh dan ekstrak seluruh dataset
            kaggle.api.dataset_download_files(
                'krishd123/log-data-for-anomaly-detection',
                path='data',
                unzip=True
            )
            print("Dataset berhasil diunduh dan diekstrak!")
        except Exception as e:
            print(f"Error saat mengunduh dataset: {str(e)}")
            print("\nPastikan Anda telah:")
            print("1. Menginstal paket kaggle: pip install kaggle")
            print("2. Menempatkan kaggle.json di direktori ~/.config/kaggle/")
            print("3. Mengatur izin yang benar: chmod 600 ~/.config/kaggle/kaggle.json")
            return False
        # Periksa lagi apakah file ada setelah ekstraksi
        if not os.path.exists(target_file_path):
            print(f"File {target_file_path} tidak ditemukan setelah ekstraksi.")
            return False
    
    print(f"File {target_file_path} berhasil ditemukan!")
    return True

def prepare_dataset():
    """
    Menyiapkan dataset untuk pelatihan dan pengujian.
    Fungsi ini akan:
    1. Membaca file log
    2. Membuat label berdasarkan pola anomali
    3. Membagi dataset menjadi set pelatihan dan pengujian
    4. Menyimpan dataset yang telah diproses
    """
    print("Menyiapkan dataset...")
    
    # Baca file log baris per baris ke dalam DataFrame
    log_file_path = os.path.join('data', 'hdfs_log', 'hdfs.log', 'sorted.log')
    with open(log_file_path, 'r', encoding='utf-8', errors='ignore') as f:
        logs = f.readlines()
    df = pd.DataFrame({'log': [log.strip() for log in logs]})
    print(f"Total entri log: {len(df)}")
    
    # Definisikan pola yang mungkin menunjukkan anomali
    anomaly_patterns = [
        'Exception',
        'Error',
        'Failed',
        'Timeout',
        'Connection refused',
        'Permission denied',
        'OutOfMemory',
        'NullPointerException',
        'StackOverflowError',
        'ClassNotFoundException',
        'WARNING',
        'ERROR',
        'CRITICAL',
        'Fatal',
        'Invalid',
        'Missing',
        'Not found',
        'Access denied',
        'Authentication failed',
        'Connection lost'
    ]
    
    # Buat label berdasarkan pola
    labels = np.zeros(len(df))
    for i, log in enumerate(df['log']):
        if any(pattern.lower() in str(log).lower() for pattern in anomaly_patterns):
            labels[i] = 1
    
    # Tambahkan label ke dataframe
    df['label'] = labels
    
    # Bagi menjadi set pelatihan dan pengujian
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Simpan dataset
    train_df.to_csv('data/train_logs.csv', index=False)
    test_df.to_csv('data/test_logs.csv', index=False)
    
    print(f"\nDataset berhasil disiapkan!")
    print(f"Ukuran set pelatihan: {len(train_df)}")
    print(f"Ukuran set pengujian: {len(test_df)}")
    print(f"Jumlah anomali dalam set pelatihan: {train_df['label'].sum()}")
    print(f"Jumlah anomali dalam set pengujian: {test_df['label'].sum()}")
    print(f"Persentase anomali dalam set pelatihan: {(train_df['label'].sum() / len(train_df) * 100):.2f}%")
    print(f"Persentase anomali dalam set pengujian: {(test_df['label'].sum() / len(test_df) * 100):.2f}%")

def main():
    if download_hdfs_dataset():
        prepare_dataset()

if __name__ == "__main__":
    main() 