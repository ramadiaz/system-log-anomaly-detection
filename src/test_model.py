import pandas as pd
import numpy as np
from preprocessing import LogPreprocessor
from model import AnomalyDetector
from utils import plot_anomalies, generate_report
import os

def test_model():
    """
    Test the anomaly detection model with the prepared dataset.
    """
    print("Loading test dataset...")
    test_df = pd.read_csv('data/test_logs.csv')
    
    # Initialize components
    preprocessor = LogPreprocessor()
    detector = AnomalyDetector()
    
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
    
    # Generate visualizations and report
    print("Generating visualizations and report...")
    plot_path = os.path.join('static', 'test_anomalies.png')
    plot_anomalies(test_df['log'].tolist(), predictions, scores, plot_path)
    
    report_path = os.path.join('reports', 'test_report.txt')
    generate_report(test_df['log'].tolist(), predictions, scores, report_path)
    
    # Calculate metrics
    true_anomalies = test_df['label'].values
    predicted_anomalies = (predictions == -1).astype(int)
    
    # Calculate accuracy
    accuracy = np.mean(true_anomalies == predicted_anomalies)
    
    # Calculate precision and recall
    true_positives = np.sum((true_anomalies == 1) & (predicted_anomalies == 1))
    false_positives = np.sum((true_anomalies == 0) & (predicted_anomalies == 1))
    false_negatives = np.sum((true_anomalies == 1) & (predicted_anomalies == 0))
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    
    print("\nTest Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"\nVisualization saved to: {plot_path}")
    print(f"Detailed report saved to: {report_path}")

if __name__ == "__main__":
    test_model() 