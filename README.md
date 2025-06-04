# System Log Anomaly Detection

This project implements a machine learning-based system for detecting anomalies in system logs. It helps identify unusual patterns and potential security threats in system logs.

## Features

- Log data preprocessing and feature extraction
- Anomaly detection using machine learning
- Real-time log monitoring
- Visualization of anomalies
- REST API for log analysis
- Synthetic data generation for larger training sets

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python app.py
```

## Project Structure

```
.
├── app.py                 # Flask application
├── models/               # ML model files
├── data/                 # Dataset and processed data
├── src/                  # Source code
│   ├── preprocessing.py  # Data preprocessing
│   ├── model.py         # ML model implementation
│   ├── utils.py         # Utility functions
│   ├── download_dataset.py  # Script to download and prepare the dataset
│   ├── train_model.py   # Script to train the model
│   └── test_model.py    # Script to test the model
├── requirements.txt      # Project dependencies
└── README.md            # Project documentation
```

## Dataset

The project uses the HDFS (Hadoop Distributed File System) log dataset as a base, which is then augmented with synthetic data to create a larger training set.

### Dataset Generation

1. **Base Dataset**: Uses HDFS_2k.log from LogPai repository
2. **Synthetic Data Generation**:
   - Creates variations of original logs
   - Adds random variations in log levels
   - Generates 10 variations per original log
   - Combines original and synthetic data

### Anomaly Detection Patterns

The system detects anomalies based on patterns like:
- Log levels (ERROR, WARNING, CRITICAL)
- Exception messages
- Error indicators
- Connection issues
- Permission problems
- Memory issues
- Timeout events

## Model

The project implements an Isolation Forest algorithm for anomaly detection, which is particularly effective for detecting anomalies in system logs.

## Usage

1. **Download and Prepare the Dataset:**
   ```bash
   python src/download_dataset.py
   ```
   This will:
   - Download the base HDFS dataset
   - Generate synthetic variations
   - Create training and test sets
   - Apply anomaly labels

2. **Train the Model:**
   ```bash
   python src/train_model.py
   ```

3. **Test the Model:**
   ```bash
   python src/test_model.py
   ```

4. **Run the Web Application:**
   ```bash
   python app.py
   ```

## Anomaly Detection

The model marks logs as anomalies based on:
- Statistical outliers in features
- Pattern matching for known error types
- Log level analysis
- Message content analysis
- Special character and number patterns

Check the `reports/test_report.txt` file for detailed anomaly reports.

## Contributing

Feel free to submit issues and enhancement requests! 