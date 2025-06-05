import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import re
from tqdm import tqdm
from datetime import datetime

class LogPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        
    def extract_features(self, log_line):
        """Extract features from a single log line."""
        features = {}
        
        # Extract timestamp
        timestamp_pattern = r'\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2}|\d{6}\s\d{6}'
        timestamp = re.search(timestamp_pattern, log_line)
        if timestamp:
            try:
                timestamp_str = timestamp.group()
                if len(timestamp_str) == 19:  # Format: YYYY-MM-DD HH:MM:SS
                    features['timestamp'] = pd.to_datetime(timestamp_str)
                else:  # Format: MMDDYY HHMMSS
                    features['timestamp'] = pd.to_datetime(timestamp_str, format='%m%d%y %H%M%S')
            except:
                features['timestamp'] = None
        
        # Extract log level
        log_levels = ['INFO', 'WARNING', 'ERROR', 'CRITICAL', 'DEBUG', 'FATAL']
        features['log_level'] = 'UNKNOWN'
        for level in log_levels:
            if level in log_line:
                features['log_level'] = level
                break
        
        # Extract message length
        features['message_length'] = len(log_line)
        
        # Extract number of special characters
        features['special_chars'] = len(re.findall(r'[^a-zA-Z0-9\s]', log_line))
        
        # Extract number of numbers
        features['numbers'] = len(re.findall(r'\d+', log_line))
        
        # Extract IP addresses
        ip_pattern = r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}'
        ip_addresses = re.findall(ip_pattern, log_line)
        features['ip_count'] = len(ip_addresses)
        
        # Extract block IDs
        block_pattern = r'blk_[-]?\d+'
        block_ids = re.findall(block_pattern, log_line)
        features['block_count'] = len(block_ids)
        
        # HDFS specific patterns
        hdfs_patterns = {
            'block_delete': r'BLOCK\* ask.*to delete',
            'block_received': r'BLOCK\* Received',
            'block_allocated': r'BLOCK\* allocate',
            'block_replicated': r'BLOCK\* replicate',
            'block_verified': r'BLOCK\* verify',
            'block_corrupt': r'BLOCK\* corrupt',
            'block_missing': r'BLOCK\* missing',
            'block_under_replicated': r'BLOCK\* under_replicated',
            'block_over_replicated': r'BLOCK\* over_replicated',
            'block_recovery': r'BLOCK\* recovery',
            'block_failed': r'BLOCK\* failed',
            'block_timeout': r'BLOCK\* timeout',
            'block_error': r'BLOCK\* error',
            'block_exception': r'BLOCK\* exception',
            'block_warning': r'BLOCK\* warning'
        }
        
        # Count HDFS specific patterns
        for pattern_name, pattern in hdfs_patterns.items():
            features[f'hdfs_{pattern_name}'] = 1 if re.search(pattern, log_line) else 0
        
        # Extract error patterns
        error_patterns = [
            'Exception', 'Error', 'Failed', 'Timeout', 'Connection refused',
            'Permission denied', 'OutOfMemory', 'NullPointerException',
            'StackOverflowError', 'ClassNotFoundException', 'IOException',
            'SocketTimeoutException', 'ConnectException', 'UnknownHostException',
            'EOFException', 'FileNotFoundException', 'SecurityException'
        ]
        features['error_count'] = sum(1 for pattern in error_patterns if pattern in log_line)
        
        # Extract word count
        features['word_count'] = len(log_line.split())
        
        # Extract unique words ratio
        words = log_line.split()
        if words:
            features['unique_words_ratio'] = len(set(words)) / len(words)
        else:
            features['unique_words_ratio'] = 0
            
        return features
    
    def preprocess_logs(self, log_data, progress_bar=False):
        """Preprocess a list of log lines."""
        # Convert logs to DataFrame
        if progress_bar:
            features_list = [self.extract_features(log) for log in tqdm(log_data, desc="Preprocessing logs")]
        else:
            features_list = [self.extract_features(log) for log in log_data]
        df = pd.DataFrame(features_list)
        
        # Handle missing values
        df = df.ffill()
        
        # Convert timestamp to numerical features
        if 'timestamp' in df.columns:
            df['hour'] = df['timestamp'].dt.hour
            df['minute'] = df['timestamp'].dt.minute
            df['second'] = df['timestamp'].dt.second
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df = df.drop('timestamp', axis=1)
        
        # Convert categorical variables
        if 'log_level' in df.columns:
            df = pd.get_dummies(df, columns=['log_level'])
        
        # Scale numerical features
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        df[numerical_cols] = self.scaler.fit_transform(df[numerical_cols])
        
        return df
    
    def save_scaler(self, path):
        """Save the scaler for later use."""
        import joblib
        joblib.dump(self.scaler, path)
    
    def load_scaler(self, path):
        """Load a saved scaler."""
        import joblib
        self.scaler = joblib.load(path) 