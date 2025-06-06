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
        
        # Extract log level (most important feature)
        log_levels = ['INFO', 'WARNING', 'ERROR', 'CRITICAL', 'DEBUG', 'FATAL']
        features['log_level'] = 'UNKNOWN'
        for level in log_levels:
            if level in log_line:
                features['log_level'] = level
                break
        
        # Extract error patterns (key indicator of anomalies)
        error_patterns = [
            'Exception', 'Error', 'Failed', 'Timeout', 'Connection refused',
            'Permission denied', 'OutOfMemory', 'NullPointerException',
            'StackOverflowError', 'ClassNotFoundException', 'WARNING',
            'ERROR', 'CRITICAL', 'Fatal', 'Invalid', 'Missing',
            'Not found', 'Access denied', 'Authentication failed',
            'Connection lost'
        ]
        
        # Count error patterns
        features['error_count'] = sum(1 for pattern in error_patterns if pattern.lower() in log_line.lower())
        
        # Extract message length (simple but effective feature)
        features['message_length'] = len(log_line)
        
        # Extract special characters (indicator of unusual patterns)
        features['special_chars'] = len(re.findall(r'[^a-zA-Z0-9\s]', log_line))
        
        # Extract numbers (indicator of system events)
        features['numbers'] = len(re.findall(r'\d+', log_line))
        
        # Extract block IDs (specific to HDFS logs)
        block_pattern = r'blk_[-]?\d+'
        features['block_count'] = len(re.findall(block_pattern, log_line))
        
        return features
    
    def preprocess_logs(self, log_data, progress_bar=False):
        """Preprocess a list of log lines."""
        # Convert logs to DataFrame
        if progress_bar:
            features_list = [self.extract_features(log) for log in tqdm(log_data, desc="Preprocessing logs")]
        else:
            features_list = [self.extract_features(log) for log in log_data]
        df = pd.DataFrame(features_list)
        
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