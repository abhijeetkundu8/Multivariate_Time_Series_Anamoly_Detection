PCA-Based Anomaly Detection

This project implements an anomaly detection pipeline using Principal Component Analysis (PCA).
It identifies abnormal behavior in multivariate time-series data and attributes anomalies to the most contributing features.

📂 Files in this Project

AnomalyDetection.ipynb → Jupyter Notebook (interactive version of the code).

AnomalyDetection.py → Python script (directly executable version).

TEP_Train_Test.csv → Input dataset.

TEP_with_anomaly_output.csv → Output file with anomaly scores and top features.

Documentation.pdf → Project explanation, methodology, and references.

⚙️ Installation

Make sure you have Python 3.10+ installed.
Install dependencies using:

pip install numpy pandas scikit-learn

▶️ How to Run
Option 1: Run the Python Script

Place the following files in the same folder:

AnomalyDetection.py

TEP_Train_Test.csv

Open a terminal/command prompt in that folder.

Run:

python AnomalyDetection.py


The script will generate:

TEP_with_anomaly_output.csv → containing anomaly scores and top features.

Option 2: Run the Jupyter Notebook

Open AnomalyDetection.ipynb in Jupyter Notebook or JupyterLab.

Upload the dataset (TEP_Train_Test.csv).

Run all cells sequentially.

The output file TEP_with_anomaly_output.csv will be generated.

📊 Output

The output CSV contains:

Abnormality_score → Score between 0–100 (higher = more abnormal).

top_feature_1 … top_feature_7 → Features that contributed most to the anomaly.

📚 References

Step-by-Step Explanation of PCA

PCA in Python – Datacamp

Anomaly Detection using PCA – Analytics Vidhya

Anomaly Detection with PCA – Visual Studio Magazine

Big Data Journal – PCA-based anomaly detection research

Kaggle: PCA-based Anomaly Detection

✅ Conclusion

This project demonstrates the effectiveness of PCA for anomaly detection in multivariate time-series datasets.
The approach successfully:

Detects abnormal patterns in industrial process data.

Assigns an interpretable Abnormality_score (0–100).

Provides feature attribution by ranking the top variables that contribute to anomalies.

This makes the method useful not only for detection but also for root-cause analysis in real-world scenarios.

🚀 Future Work

Extend the pipeline to support real-time anomaly detection on streaming data.

Compare PCA with other anomaly detection techniques (Isolation Forest, Autoencoders, LSTMs).

Optimize feature attribution methods to improve interpretability.

Deploy as an API or dashboard for interactive anomaly monitoring.

Test the framework on more diverse industrial datasets for generalization.
