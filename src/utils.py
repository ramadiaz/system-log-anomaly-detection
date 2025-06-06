import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np

def load_log_data(file_path):
    """
    Load log data from a file.
    
    Args:
        file_path (str): Path to the log file
        
    Returns:
        list: List of log lines
    """
    with open(file_path, 'r') as f:
        return f.readlines()

def plot_anomalies(log_data, predictions, scores, save_path=None):
    """
    Plot the anomalies in the log data.
    
    Args:
        log_data (list): List of log lines
        predictions (numpy.ndarray): Model predictions (1 for anomalies, 0 for normal)
        scores (numpy.ndarray): Anomaly scores
        save_path (str, optional): Path to save the plot
    """
    plt.figure(figsize=(15, 6))
    
    # Plot anomaly scores
    plt.subplot(1, 2, 1)
    plt.plot(scores, label='Anomaly Score')
    plt.axhline(y=np.mean(scores) + 2*np.std(scores), color='r', linestyle='--', label='Threshold')
    plt.title('Anomaly Scores')
    plt.xlabel('Log Entry')
    plt.ylabel('Score')
    plt.legend()
    
    # Plot predictions
    plt.subplot(1, 2, 2)
    plt.scatter(range(len(predictions)), predictions, c=predictions, cmap='coolwarm')
    plt.title('Anomaly Predictions')
    plt.xlabel('Log Entry')
    plt.ylabel('Prediction (1: Anomaly, 0: Normal)')
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def generate_report(log_data, predictions, scores, output_path):
    """
    Generate a detailed report of the anomalies.
    
    Args:
        log_data (list): List of log lines
        predictions (numpy.ndarray): Model predictions (1 for anomalies, 0 for normal)
        scores (numpy.ndarray): Anomaly scores
        output_path (str): Path to save the report
    """
    anomalies = []
    for i, (log, pred, score) in enumerate(zip(log_data, predictions, scores)):
        if pred == 1:  # Anomaly
            anomalies.append({
                'index': i,
                'log': log.strip(),
                'score': score
            })
    
    # Sort anomalies by score
    anomalies.sort(key=lambda x: x['score'], reverse=True)
    
    # Generate report
    with open(output_path, 'w') as f:
        f.write(f"Anomaly Detection Report\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total logs analyzed: {len(log_data)}\n")
        f.write(f"Number of anomalies detected: {len(anomalies)}\n\n")
        
        f.write("Top Anomalies:\n")
        f.write("-" * 80 + "\n")
        for anomaly in anomalies[:10]:  # Show top 10 anomalies
            f.write(f"Index: {anomaly['index']}\n")
            f.write(f"Score: {anomaly['score']:.4f}\n")
            f.write(f"Log: {anomaly['log']}\n")
            f.write("-" * 80 + "\n") 