import numpy as np
from sklearn.ensemble import IsolationForest
import joblib
import os

class AnomalyDetector:
    def __init__(self, contamination=0.1):
        """
        Initialize the anomaly detector.
        
        Args:
            contamination (float): The proportion of outliers in the data set.
        """
        self.model = IsolationForest(
            n_estimators=100,
            max_samples='auto',
            contamination=contamination,
            random_state=42
        )
        
    def train(self, X):
        """
        Train the anomaly detection model.
        
        Args:
            X (numpy.ndarray): Training data
        """
        self.model.fit(X)
        
    def predict(self, X):
        """
        Predict anomalies in the data.
        
        Args:
            X (numpy.ndarray): Data to predict
            
        Returns:
            numpy.ndarray: -1 for anomalies, 1 for normal
        """
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Get anomaly scores for the data.
        
        Args:
            X (numpy.ndarray): Data to predict
            
        Returns:
            numpy.ndarray: Anomaly scores (higher means more anomalous)
        """
        return -self.model.score_samples(X)
    
    def save_model(self, path):
        """
        Save the model to disk.
        
        Args:
            path (str): Path to save the model
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)
    
    def load_model(self, path):
        """
        Load a saved model from disk.
        
        Args:
            path (str): Path to the saved model
        """
        self.model = joblib.load(path) 