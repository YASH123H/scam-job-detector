# 🕵️‍♂️ Fake Job Posting Detector

This tool detects potentially fraudulent job listings using a machine learning classifier trained on real-world data.

---

## 🚀 Features

- Upload CSV with job listings
- Get predictions + fraud probabilities
- Visual dashboard with charts:
- Histogram of fraud probability
- Bar and Pie chart of legit vs fraudulent
- Top 10 most suspicious listings
- Title length vs fraud probability scatter
- Download prediction results
- Model performance (F1-score + CV) shown in notebook

---

## 🧠 Model

- Random Forest Classifier (n=100)
- TF-IDF vectorization of combined text
- 5-fold Cross-Validation
- F1-Score: `~0.71` (update with your actual mean)

---

## 🗂️ Files

| File | Description |
|------|-------------|
| `app.py` | Streamlit frontend |
| `scam_job_detection.ipynb` | Model training & evaluation |
| `fraud_job_model.pkl` | Saved model |
| `README.md` | Project info |
| `requirements.txt` | Dependencies |

---