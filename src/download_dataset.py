import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import requests
import zipfile
import io

def download_hdfs_dataset():
    """
    Download the HDFS log dataset from Kaggle.
    Note: You need to have your Kaggle API credentials set up.
    """
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Download the dataset
    print("Downloading HDFS dataset...")
    url = "https://raw.githubusercontent.com/logpai/loghub/master/HDFS/HDFS_2k.log"
    response = requests.get(url)
    
    if response.status_code == 200:
        # Save the raw log file
        with open('data/HDFS_2k.log', 'w') as f:
            f.write(response.text)
        print("Dataset downloaded successfully!")
    else:
        print(f"Failed to download dataset. Status code: {response.status_code}")
        return False
    
    return True

def prepare_dataset():
    """
    Prepare the dataset for training and testing.
    """
    print("Preparing dataset...")
    
    # Read the log file
    with open('data/HDFS_2k.log', 'r') as f:
        logs = f.readlines()
    
    # Create a simple labeled dataset (for demonstration)
    # In a real scenario, you would use the actual labels from the dataset
    np.random.seed(42)
    labels = np.random.choice([0, 1], size=len(logs), p=[0.95, 0.05])  # 5% anomalies
    
    # Create DataFrame
    df = pd.DataFrame({
        'log': logs,
        'label': labels
    })
    
    # Split into train and test sets
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Save the datasets
    train_df.to_csv('data/train_logs.csv', index=False)
    test_df.to_csv('data/test_logs.csv', index=False)
    
    print(f"Dataset prepared successfully!")
    print(f"Training set size: {len(train_df)}")
    print(f"Test set size: {len(test_df)}")
    print(f"Number of anomalies in training set: {train_df['label'].sum()}")
    print(f"Number of anomalies in test set: {test_df['label'].sum()}")

def main():
    if download_hdfs_dataset():
        prepare_dataset()

if __name__ == "__main__":
    main() 