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
    Generate a detailed report listing all detected anomalies with their scores in CSV format.
    
    Args:
        logs (list): List of log entries
        predictions (array): Array of anomaly predictions (0 or 1)
        scores (array): Array of anomaly scores
        output_path (str): Path to save the anomaly list report
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
        f.write(f"Anomaly Detection Report\n")
        f.write(f"Generated on: {timestamp}\n")
        f.write(f"Total anomalies detected: {len(anomaly_indices)}\n\n")
    
    # Append the DataFrame to the same file
    df.to_csv(output_path, mode='a', index=False)

def test_model():
    """
    Test the anomaly detection model with the prepared dataset.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    print("Loading test dataset...")
    test_df = pd.read_csv('data/test_logs.csv')
    print(f"Total test entries: {len(test_df)}")
    print(f"Actual anomalies in test set: {test_df['label'].sum()}")
    
    # Calculate contamination based on actual anomaly ratio
    anomaly_ratio = test_df['label'].mean()
    print(f"Anomaly ratio in dataset: {anomaly_ratio:.4f}")
    
    # Initialize components with calculated contamination
    preprocessor = LogPreprocessor()
    detector = AnomalyDetector(
        contamination=max(anomaly_ratio, 0.1),  # Use actual ratio or minimum 0.1
        threshold_percentile=50
    )
    
    # Load trained model and scaler
    print("Loading trained model and scaler...")
    try:
        detector.load_model('models/anomaly_detector.joblib')
        preprocessor.load_scaler('models/scaler.joblib')
    except FileNotFoundError:
        print("Error: Model files not found. Please train the model first using:")
        print("python src/train_model.py")
        return
    
    # Process the test data
    print("Processing test data...")
    processed_data = preprocessor.preprocess_logs(test_df['log'].tolist())
    
    # Make predictions
    print("Making predictions...")
    predictions = detector.predict(processed_data)
    scores = detector.predict_proba(processed_data)
    
    # Generate reports
    print("Generating reports...")
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
    
    print("\nTest Results:")
    print(f"Total logs analyzed: {len(test_df)}")
    print(f"Actual anomalies in dataset: {true_anomalies.sum()}")
    print(f"Detected anomalies: {predicted_anomalies.sum()}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"\nDetailed report saved to: {report_path}")
    print(f"Anomaly list saved to: {anomaly_list_path}")

if __name__ == "__main__":
    test_model() 