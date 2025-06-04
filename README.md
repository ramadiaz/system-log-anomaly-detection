# System Log Anomaly Detection

This project implements a machine learning-based system for detecting anomalies in system logs. It helps identify unusual patterns and potential security threats in system logs.

## Features

- Log data preprocessing and feature extraction
- Anomaly detection using machine learning
- Real-time log monitoring
- Visualization of anomalies
- REST API for log analysis

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

The project uses the HDFS (Hadoop Distributed File System) log dataset, which contains system logs with labeled anomalies. The dataset is available on Kaggle.

### Dataset Options

- **HDFS Full LogHub Dataset (from LogPai)**
  - [HDFS.log](https://raw.githubusercontent.com/logpai/loghub/master/HDFS/HDFS.log) (about 1.6 million lines)
  - Contains real anomaly labels.

- **Kaggle: HDFS Log Anomaly Detection Dataset**
  - [Kaggle HDFS Log Anomaly Detection Dataset](https://www.kaggle.com/datasets/cheongwoongkang/hdfs-log-anomaly-detection)
  - Contains both log lines and anomaly labels.

## Model

The project implements an Isolation Forest algorithm for anomaly detection, which is particularly effective for detecting anomalies in system logs.

## Usage

1. **Download and Prepare the Dataset:**
   ```bash
   python src/download_dataset.py
   ```

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

The model marks logs as anomalies based on statistical outliers in features like:
- Log level (ERROR, WARNING, CRITICAL vs. INFO)
- Message length
- Number of special characters
- Number of numbers
- Rare words (e.g., Exception, failure, crash)

Check the `reports/test_report.txt` file for detailed anomaly reports.

## Contributing

Feel free to submit issues and enhancement requests! 