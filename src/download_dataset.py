import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import kaggle
import zipfile
import io

def download_hdfs_dataset():
    """
    Download the log dataset from Kaggle using the Kaggle API.
    Note: You need to have kaggle.json in ~/.kaggle/ directory
    """
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Check if the target file exists
    target_file_path = os.path.join('data', 'hdfs_log', 'hdfs.log', 'sorted.log')
    if not os.path.exists(target_file_path):
        print("Downloading dataset from Kaggle...")
        try:
            # Download and unzip the entire dataset
            kaggle.api.dataset_download_files(
                'krishd123/log-data-for-anomaly-detection',
                path='data',
                unzip=True
            )
            print("Dataset downloaded and extracted successfully!")
        except Exception as e:
            print(f"Error downloading dataset: {str(e)}")
            print("\nPlease make sure you have:")
            print("1. Installed kaggle package: pip install kaggle")
            print("2. Placed kaggle.json in ~/.config/kaggle/ directory")
            print("3. Set correct permissions: chmod 600 ~/.config/kaggle/kaggle.json")
            return False
        # Check again if the file exists after extraction
        if not os.path.exists(target_file_path):
            print(f"File {target_file_path} not found after extraction.")
            return False
    
    print(f"File {target_file_path} found successfully!")
    return True

def prepare_dataset():
    """
    Prepare the dataset for training and testing.
    """
    print("Preparing dataset...")
    
    # Read the log file line by line into a DataFrame
    log_file_path = os.path.join('data', 'hdfs_log', 'hdfs.log', 'sorted.log')
    with open(log_file_path, 'r', encoding='utf-8', errors='ignore') as f:
        logs = f.readlines()
    df = pd.DataFrame({'log': [log.strip() for log in logs]})
    print(f"Total log entries: {len(df)}")
    
    # Define patterns that might indicate anomalies
    anomaly_patterns = [
        'Exception',
        'Error',
        'Failed',
        'Timeout',
        'Connection refused',
        'Permission denied',
        'OutOfMemory',
        'NullPointerException',
        'StackOverflowError',
        'ClassNotFoundException',
        'WARNING',
        'ERROR',
        'CRITICAL',
        'Fatal',
        'Invalid',
        'Missing',
        'Not found',
        'Access denied',
        'Authentication failed',
        'Connection lost'
    ]
    
    # Create labels based on patterns
    labels = np.zeros(len(df))
    for i, log in enumerate(df['log']):
        if any(pattern.lower() in str(log).lower() for pattern in anomaly_patterns):
            labels[i] = 1
    
    # Add the labels to the dataframe
    df['label'] = labels
    
    # Split into train and test sets
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Save the datasets
    train_df.to_csv('data/train_logs.csv', index=False)
    test_df.to_csv('data/test_logs.csv', index=False)
    
    print(f"\nDataset prepared successfully!")
    print(f"Training set size: {len(train_df)}")
    print(f"Test set size: {len(test_df)}")
    print(f"Number of anomalies in training set: {train_df['label'].sum()}")
    print(f"Number of anomalies in test set: {test_df['label'].sum()}")
    print(f"Anomaly percentage in training set: {(train_df['label'].sum() / len(train_df) * 100):.2f}%")
    print(f"Anomaly percentage in test set: {(test_df['label'].sum() / len(test_df) * 100):.2f}%")

def main():
    if download_hdfs_dataset():
        prepare_dataset()

if __name__ == "__main__":
    main() 