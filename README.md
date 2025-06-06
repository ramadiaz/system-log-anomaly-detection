# System Log Anomaly Detection

This project implements a machine learning-based system for detecting anomalies in system logs using the Isolation Forest algorithm. The system processes log data, extracts meaningful features, and identifies unusual patterns that may indicate system issues or security threats.

## Features

- Log data preprocessing and feature extraction
- Anomaly detection using Isolation Forest algorithm
- CSV-based anomaly reporting
- Comprehensive feature engineering for log analysis

## System Requirements

- Python 3.8 or newer
- pip (Python package manager)
- Internet access for dataset download
- Kaggle API credentials

## Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up Kaggle API:
   - Install kaggle package: `pip install kaggle`
   - Place kaggle.json in `~/.config/kaggle/`
   - Set correct permissions: `chmod 600 ~/.config/kaggle/kaggle.json`

## Project Structure

```
.
├── models/                   # ML model files
├── data/                     # Dataset and processed data
├── reports/                  # Analysis results and reports
├── src/                      # Source code
│   ├── preprocessing.py      # Log data preprocessing
│   ├── model.py             # ML model implementation
│   ├── utils.py             # Utility functions
│   ├── download_dataset.py  # Dataset download and preparation
│   ├── train_model.py       # Model training script
│   └── test_model.py        # Model testing script
├── requirements.txt          # Project dependencies
└── README.md                # Project documentation
```

## Dataset

The project uses the HDFS (Hadoop Distributed File System) log dataset from Kaggle. The dataset is processed to identify anomalies based on predefined patterns and split into training and testing sets.

### Dataset Processing

1. **Source**: HDFS log dataset from Kaggle
2. **Processing Steps**:
   - Download dataset using Kaggle API
   - Extract and read log entries
   - Label anomalies based on predefined patterns
   - Split into training (80%) and testing (20%) sets

### Feature Engineering

The system extracts the following features from log entries:
- Log levels (INFO, WARNING, ERROR, CRITICAL, DEBUG, FATAL)
- Error pattern counts
- Message length
- Special character count
- Number count
- Block ID count (HDFS-specific)

## Model

The project implements an Isolation Forest algorithm for anomaly detection with the following characteristics:
- 100 estimators (trees) for robust detection
- Adaptive contamination rate based on dataset
- Dynamic threshold calculation using percentile and standard deviation
- Parallel processing support (4 cores)

## Usage

1. **Download and Prepare Dataset:**
   ```bash
   python src/download_dataset.py
   ```
   This will:
   - Download the HDFS dataset from Kaggle
   - Process and label the log entries
   - Create training and test sets
   - Save processed datasets

2. **Train Model:**
   ```bash
   python src/train_model.py
   ```

3. **Test Model:**
   ```bash
   python src/test_model.py
   ```

## Anomaly Detection

The model identifies anomalies based on:
- Statistical outliers in features
- Pattern matching for known error types
- Log level analysis
- Message content analysis
- Special character and number patterns

The anomaly report is available in CSV format at `reports/anomaly_list.csv` with:
- Log index
- Log entry
- Anomaly score

## Performance Metrics

The system provides the following evaluation metrics:
- Accuracy
- Precision
- Recall
- Detailed anomaly reports
- Comprehensive test results

## Contributing

Feel free to submit issues and enhancement requests!