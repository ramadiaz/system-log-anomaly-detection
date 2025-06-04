from flask import Flask, request, jsonify, render_template
import os
from src.preprocessing import LogPreprocessor
from src.model import AnomalyDetector
from src.utils import load_log_data, plot_anomalies, generate_report
import numpy as np

app = Flask(__name__)

# Initialize components
preprocessor = LogPreprocessor()
detector = AnomalyDetector()

# Create necessary directories
os.makedirs('static', exist_ok=True)
os.makedirs('reports', exist_ok=True)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_logs():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Save uploaded file
    file_path = os.path.join('data', 'uploaded_logs.txt')
    os.makedirs('data', exist_ok=True)
    file.save(file_path)
    
    # Process logs
    log_data = load_log_data(file_path)
    processed_data = preprocessor.preprocess_logs(log_data)
    
    # Detect anomalies
    predictions = detector.predict(processed_data)
    scores = detector.predict_proba(processed_data)
    
    # Generate visualizations and report
    plot_path = os.path.join('static', 'anomalies.png')
    plot_anomalies(log_data, predictions, scores, plot_path)
    
    report_path = os.path.join('reports', 'anomaly_report.txt')
    generate_report(log_data, predictions, scores, report_path)
    
    # Prepare response
    anomalies = []
    for i, (log, pred, score) in enumerate(zip(log_data, predictions, scores)):
        if pred == -1:  # Anomaly
            anomalies.append({
                'index': i,
                'log': log.strip(),
                'score': float(score)
            })
    
    return jsonify({
        'total_logs': len(log_data),
        'anomalies_detected': len(anomalies),
        'anomalies': anomalies[:10],  # Return top 10 anomalies
        'plot_url': '/static/anomalies.png',
        'report_url': '/reports/anomaly_report.txt'
    })

@app.route('/train', methods=['POST'])
def train_model():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Save training file
    file_path = os.path.join('data', 'training_logs.txt')
    os.makedirs('data', exist_ok=True)
    file.save(file_path)
    
    # Process and train
    log_data = load_log_data(file_path)
    processed_data = preprocessor.preprocess_logs(log_data)
    detector.train(processed_data)
    
    # Save model and scaler
    detector.save_model('models/anomaly_detector.joblib')
    preprocessor.save_scaler('models/scaler.joblib')
    
    return jsonify({'message': 'Model trained successfully'})

if __name__ == '__main__':
    app.run(debug=True) 