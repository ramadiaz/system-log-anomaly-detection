import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import re

class LogPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        
    def extract_features(self, log_line):
        """Extract features from a single log line."""
        features = {}
        
        # Extract timestamp
        timestamp_pattern = r'\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2}'
        timestamp = re.search(timestamp_pattern, log_line)
        if timestamp:
            features['timestamp'] = pd.to_datetime(timestamp.group())
        
        # Extract log level
        log_levels = ['INFO', 'WARNING', 'ERROR', 'CRITICAL', 'DEBUG']
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
        
        return features
    
    def preprocess_logs(self, log_data):
        """Preprocess a list of log lines."""
        # Convert logs to DataFrame
        features_list = [self.extract_features(log) for log in log_data]
        df = pd.DataFrame(features_list)
        
        # Handle missing values
        df = df.ffill()
        
        # Convert timestamp to numerical features
        if 'timestamp' in df.columns:
            df['hour'] = df['timestamp'].dt.hour
            df['minute'] = df['timestamp'].dt.minute
            df['second'] = df['timestamp'].dt.second
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