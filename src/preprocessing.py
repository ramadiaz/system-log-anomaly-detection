import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import re
from tqdm import tqdm
from datetime import datetime

class LogPreprocessor:
    def __init__(self):
        """
        Inisialisasi preprocessor log.
        """
        self.scaler = StandardScaler()
        
    def extract_features(self, log_line):
        """
        Mengekstrak fitur dari satu baris log.
        
        Parameter:
            log_line (str): Baris log yang akan diproses
            
        Returns:
            dict: Kamus berisi fitur-fitur yang diekstrak
        """
        features = {}
        
        # Ekstrak level log (fitur terpenting)
        log_levels = ['INFO', 'WARNING', 'ERROR', 'CRITICAL', 'DEBUG', 'FATAL']
        features['log_level'] = 'UNKNOWN'
        for level in log_levels:
            if level in log_line:
                features['log_level'] = level
                break
        
        # Ekstrak pola error (indikator utama anomali)
        error_patterns = [
            'Exception', 'Error', 'Failed', 'Timeout', 'Connection refused',
            'Permission denied', 'OutOfMemory', 'NullPointerException',
            'StackOverflowError', 'ClassNotFoundException', 'WARNING',
            'ERROR', 'CRITICAL', 'Fatal', 'Invalid', 'Missing',
            'Not found', 'Access denied', 'Authentication failed',
            'Connection lost'
        ]
        
        # Hitung pola error
        features['error_count'] = sum(1 for pattern in error_patterns if pattern.lower() in log_line.lower())
        
        # Ekstrak panjang pesan (fitur sederhana tapi efektif)
        features['message_length'] = len(log_line)
        
        # Ekstrak karakter khusus (indikator pola tidak biasa)
        features['special_chars'] = len(re.findall(r'[^a-zA-Z0-9\s]', log_line))
        
        # Ekstrak angka (indikator kejadian sistem)
        features['numbers'] = len(re.findall(r'\d+', log_line))
        
        # Ekstrak ID blok (spesifik untuk log HDFS)
        block_pattern = r'blk_[-]?\d+'
        features['block_count'] = len(re.findall(block_pattern, log_line))
        
        return features

    def preprocess_logs(self, log_data, progress_bar=False):
        """
        Memproses daftar baris log.
        """
        # Konversi log ke DataFrame
        if progress_bar:
            features_list = [self.extract_features(log) for log in tqdm(log_data, desc="Memproses log")]
        else:
            features_list = [self.extract_features(log) for log in log_data]
        df = pd.DataFrame(features_list)

        # Map log levels to anomaly scores
        anomaly_score_map = {
            'FATAL': 5,
            'CRITICAL': 4,
            'ERROR': 3,
            'WARNING': 2,
            'INFO': 1,
            'DEBUG': 0,
            'UNKNOWN': -1
        }
        df['anomaly_score'] = df['log_level'].map(anomaly_score_map)

        # Konversi variabel kategorikal
        if 'log_level' in df.columns:
            df = pd.get_dummies(df, columns=['log_level'])

        # Skala fitur numerik
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        df[numerical_cols] = self.scaler.fit_transform(df[numerical_cols])

        return df

    def save_scaler(self, path):
        """
        Menyimpan scaler untuk penggunaan selanjutnya.
        
        Parameter:
            path (str): Path untuk menyimpan scaler
        """
        import joblib
        joblib.dump(self.scaler, path)
    
    def load_scaler(self, path):
        """
        Memuat scaler yang tersimpan.
        
        Parameter:
            path (str): Path ke scaler yang tersimpan
        """
        import joblib
        self.scaler = joblib.load(path) 