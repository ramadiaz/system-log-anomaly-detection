import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np

def load_log_data(file_path):
    """
    Memuat data log dari file.
    
    Parameter:
        file_path (str): Path ke file log
        
    Returns:
        list: Daftar baris log
    """
    with open(file_path, 'r') as f:
        return f.readlines()

def plot_anomalies(log_data, predictions, scores, save_path=None):
    """
    Membuat visualisasi anomali dalam data log.
    
    Parameter:
        log_data (list): Daftar baris log
        predictions (numpy.ndarray): Prediksi model (1 untuk anomali, 0 untuk normal)
        scores (numpy.ndarray): Skor anomali
        save_path (str, optional): Path untuk menyimpan visualisasi
    """
    plt.figure(figsize=(15, 6))
    
    # Plot skor anomali
    plt.subplot(1, 2, 1)
    plt.plot(scores, label='Skor Anomali')
    plt.axhline(y=np.mean(scores) + 2*np.std(scores), color='r', linestyle='--', label='Ambang Batas')
    plt.title('Skor Anomali')
    plt.xlabel('Entri Log')
    plt.ylabel('Skor')
    plt.legend()
    
    # Plot prediksi
    plt.subplot(1, 2, 2)
    plt.scatter(range(len(predictions)), predictions, c=predictions, cmap='coolwarm')
    plt.title('Prediksi Anomali')
    plt.xlabel('Entri Log')
    plt.ylabel('Prediksi (1: Anomali, 0: Normal)')
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def generate_report(log_data, predictions, scores, output_path):
    """
    Menghasilkan laporan detail tentang anomali yang terdeteksi.
    
    Parameter:
        log_data (list): Daftar baris log
        predictions (numpy.ndarray): Prediksi model (1 untuk anomali, 0 untuk normal)
        scores (numpy.ndarray): Skor anomali
        output_path (str): Path untuk menyimpan laporan
    """
    anomalies = []
    for i, (log, pred, score) in enumerate(zip(log_data, predictions, scores)):
        if pred == 1:  # Anomali
            anomalies.append({
                'index': i,
                'log': log.strip(),
                'score': score
            })
    
    # Urutkan anomali berdasarkan skor
    anomalies.sort(key=lambda x: x['score'], reverse=True)
    
    # Buat laporan
    with open(output_path, 'w') as f:
        f.write(f"Laporan Deteksi Anomali\n")
        f.write(f"Dibuat pada: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total log yang dianalisis: {len(log_data)}\n")
        f.write(f"Jumlah anomali terdeteksi: {len(anomalies)}\n\n")
        
        f.write("Anomali Teratas:\n")
        f.write("-" * 80 + "\n")
        for anomaly in anomalies[:10]:  # Tampilkan 10 anomali teratas
            f.write(f"Index: {anomaly['index']}\n")
            f.write(f"Skor: {anomaly['score']:.4f}\n")
            f.write(f"Log: {anomaly['log']}\n")
            f.write("-" * 80 + "\n") 