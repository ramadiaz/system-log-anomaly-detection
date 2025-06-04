import pandas as pd
from preprocessing import LogPreprocessor
from model import AnomalyDetector
import os

def train_model():
    """
    Train the anomaly detection model with the prepared dataset.
    """
    print("Loading training dataset...")
    train_df = pd.read_csv('data/train_logs.csv')
    
    # Initialize components
    preprocessor = LogPreprocessor()
    detector = AnomalyDetector()
    
    # Process the training data
    print("Processing training data...")
    processed_data = preprocessor.preprocess_logs(train_df['log'].tolist())
    
    # Train the model
    print("Training model...")
    detector.train(processed_data)
    
    # Save the model and scaler
    print("Saving model and scaler...")
    os.makedirs('models', exist_ok=True)
    detector.save_model('models/anomaly_detector.joblib')
    preprocessor.save_scaler('models/scaler.joblib')
    
    print("Training completed successfully!")
    print(f"Model saved to: models/anomaly_detector.joblib")
    print(f"Scaler saved to: models/scaler.joblib")

if __name__ == "__main__":
    train_model() 