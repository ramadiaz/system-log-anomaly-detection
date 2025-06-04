import pandas as pd
from preprocessing import LogPreprocessor
from model import AnomalyDetector
import os
import logging
import time
from tqdm import tqdm

def train_model():
    """
    Train the anomaly detection model with the prepared dataset.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    start_time = time.time()
    logging.info("Loading training dataset...")
    train_df = pd.read_csv('data/train_logs.csv')
    logging.info(f"Loaded {len(train_df)} log entries.")
    
    # Initialize components
    preprocessor = LogPreprocessor()
    detector = AnomalyDetector()
    
    # Process the training data
    logging.info("Processing training data (feature extraction and scaling)...")
    t0 = time.time()
    # Use tqdm to show progress bar for preprocessing
    processed_data = preprocessor.preprocess_logs(train_df['log'].tolist(), progress_bar=True)
    logging.info(f"Processed data shape: {processed_data.shape}")
    logging.info(f"Preprocessing completed in {time.time() - t0:.2f} seconds.")
    
    # Train the model
    logging.info("Training model...")
    t0 = time.time()
    detector.train(processed_data)
    logging.info(f"Model training completed in {time.time() - t0:.2f} seconds.")
    
    # Save the model and scaler
    logging.info("Saving model and scaler...")
    os.makedirs('models', exist_ok=True)
    detector.save_model('models/anomaly_detector.joblib')
    preprocessor.save_scaler('models/scaler.joblib')
    logging.info("Model and scaler saved successfully.")
    logging.info(f"Model saved to: models/anomaly_detector.joblib")
    logging.info(f"Scaler saved to: models/scaler.joblib")
    logging.info(f"Total training pipeline completed in {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    train_model() 