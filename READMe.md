# 🕵️‍♂️ Fake Job Posting Detector

A web app that detects potentially fraudulent job listings using a trained machine learning model.

---

## 🚀 Features

- 📂 Upload CSV file with job listings
- 🔍 Predict whether each listing is **legit** or **fraudulent**
- 📊 Interactive visualizations:
  - Histogram of fraud probabilities
  - Bar chart and pie chart of legit vs fraudulent listings
  - Top 10 most suspicious job titles
  - Scatter plot of title length vs fraud probability
- 💾 Download the prediction results
- 📈 Model performance metrics (F1-Score & Cross-Validation) shown in the notebook

---

## 🧠 Model Details

- **Model**: Random Forest Classifier (`n_estimators=100`)
- **Text Features**: Combined text vectorized using TF-IDF
- **Training**: 5-fold Cross-Validation
- **F1-Score**: ~`0.71` (Update with your exact average score)

---

## 🗂️ Project Structure

| File | Description |
|------|-------------|
| `app.py` | Streamlit frontend for interaction |
| `predict_api.py` | FastAPI backend for real-time predictions |
| `fraud_job_model.pkl` | Trained ML model file |
| `scam_job_detection.ipynb` | Notebook with training and evaluation |
| `requirements.txt` | All required Python packages |
| `.env` | Environment variables (not uploaded to GitHub) |
| `README.md` | Project documentation |

---

## 🛠️ Setup & Run

### 🔧 Install Requirements
```bash
pip install -r requirements.txt
