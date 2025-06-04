import numpy as np
from sklearn.ensemble import IsolationForest
import joblib
import os

class AnomalyDetector:
    def __init__(self, contamination=0.05, threshold_percentile=95):
        """
        Initialize the anomaly detector.
        
        Args:
            contamination (float): The proportion of outliers in the data set.
            threshold_percentile (float): Percentile to use as threshold for anomaly scores.
        """
        self.model = IsolationForest(
            n_estimators=200,  # Increased from 100
            max_samples='auto',
            contamination=contamination,
            random_state=42,
            n_jobs=-1,  # Use all available cores
            verbose=1  # Show progress during training
        )
        self.threshold_percentile = threshold_percentile
        self.threshold = None
        
    def train(self, X):
        """
        Train the anomaly detection model.
        
        Args:
            X (numpy.ndarray): Training data
        """
        # Fit the model
        self.model.fit(X)
        
        # Calculate anomaly scores for training data
        scores = -self.model.score_samples(X)
        
        # Set threshold based on percentile
        self.threshold = np.percentile(scores, self.threshold_percentile)
        print(f"Anomaly threshold set to: {self.threshold:.4f}")
        
    def predict(self, X):
        """
        Predict anomalies in the data.
        
        Args:
            X (numpy.ndarray): Data to predict
            
        Returns:
            numpy.ndarray: 1 for anomalies, 0 for normal
        """
        scores = -self.model.score_samples(X)
        return (scores > self.threshold).astype(int)
    
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
        model_data = {
            'model': self.model,
            'threshold': self.threshold,
            'threshold_percentile': self.threshold_percentile
        }
        joblib.dump(model_data, path)
    
    def load_model(self, path):
        """
        Load a saved model from disk.
        
        Args:
            path (str): Path to the saved model
        """
        model_data = joblib.load(path)
        self.model = model_data['model']
        self.threshold = model_data['threshold']
        self.threshold_percentile = model_data['threshold_percentile'] 