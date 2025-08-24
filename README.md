# PCA-Based Anomaly Detection  

This project implements an anomaly detection pipeline using **Principal Component Analysis (PCA)**.  
It identifies abnormal behavior in **multivariate time-series data** and attributes anomalies to the most contributing features.  

---

## 📂 Files in this Project
- **AnomalyDetection.ipynb** → Jupyter Notebook (interactive version).  
- **AnomalyDetection.py** → Python script (standalone executable).  
- **TEP_Train_Test.csv** → Input dataset.  
- **TEP_with_anomaly_output.csv** → Output file with anomaly scores and top features.  
- **Documentation.pdf** → Project explanation, methodology, and references.  

---

## ⚙️ Installation
1. Make sure you have **Python 3.10+** installed.  
2. Install dependencies using:  
   ```bash
   pip install numpy pandas scikit-learn

  ---

  ## How to Run
**Option 1: Run the Python Script**

1.Place AnomalyDetection.py and TEP_Train_Test.csv in the same folder.

2.Open a terminal or command prompt in that folder.

Run the script:

```bash
python AnomalyDetection.py
``` 


3.The output file TEP_with_anomaly_output.csv will be generated.

**Option 2: Run the Jupyter Notebook**

1.Open Jupyter Notebook or JupyterLab.

2.Upload and open AnomalyDetection.ipynb.

3.Upload the dataset TEP_Train_Test.csv.

4.Run all cells sequentially.

5.The output file TEP_with_anomaly_output.csv will be generated.
## Conclusion

The PCA-based anomaly detection model successfully identifies unusual behavior in the dataset.

It assigns an abnormality score to each timestamp and highlights the top contributing features.

This approach provides both accuracy (PCA reconstruction error + Z-score) and explainability (feature attribution).

---

  ## 🚀 Future Work

Integration with real-time streaming data for online anomaly detection.

Development of a visualization dashboard for anomaly trends and feature contributions.

Hybrid approaches combining PCA with clustering or deep learning for higher accuracy.

Enhancements for large-scale datasets and distributed computation.

Automated hyperparameter tuning for PCA components and weightings.
