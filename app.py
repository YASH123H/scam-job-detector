# app.py

import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import gdown
import os
from email_alert import send_email_alert

# Set API URL
API_URL = "http://127.0.0.1:8001/predict" # Make sure FastAPI is running on this port

# Page settings
st.set_page_config(page_title="Fake Job Posting Detector", layout="wide")
st.image("banner.jpg", use_container_width=True)
st.title("Fake Job Posting Detector")

# Tabs
tab1, tab2 = st.tabs(["🔍 Upload & Predict", "📊 About App"])

def download_csv_from_drive(drive_url, output_path="downloaded_drive_file.csv"):
    try:
        if "id=" in drive_url:
            file_id = drive_url.split("id=")[1].split("&")[0]
        elif "/file/d/" in drive_url:
            file_id = drive_url.split("/file/d/")[1].split("/")[0]
        else:
            st.error("❌ Invalid Google Drive link format.")
            return None

        download_url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(download_url, output_path, quiet=False)
        return output_path
    except Exception as e:
        st.error(f"❌ Failed to download from Drive: {e}")
        return None

with tab1:
    st.write("""
    Upload a CSV file with job postings.  
    The app will predict the probability of each posting being fraudulent.
    """)

    uploaded_file = st.file_uploader("Choose a CSV file (Max 200MB)", type="csv")
    google_drive_url = st.text_input("Or paste a Google Drive shareable link (for large files):", key="drive_link_input")

    df = None
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
    elif google_drive_url:
        st.info("📥 Downloading file from Google Drive...")
        downloaded_path = download_csv_from_drive(google_drive_url)
        if downloaded_path:
            df = pd.read_csv(downloaded_path)

    if df is not None:
        user_email = st.text_input(
            "Enter your email address (optional for alerts)",
            placeholder="you@example.com",
            key="user_email_input"
        )

        # Fill missing values
        df['title'] = df['title'].fillna('')
        df['company_profile'] = df['company_profile'].fillna('')
        df['description'] = df['description'].fillna('')
        df['requirements'] = df['requirements'].fillna('')

        predictions = []
        probabilities = []

        st.write("🔍 Running Predictions...")

        for index, row in df.iterrows():
            job_data = {
                "title": row['title'],
                "company_profile": row['company_profile'],
                "description": row['description'],
                "requirements": row['requirements']
            }

            try:
                response = requests.post(API_URL, json=job_data)
                if response.status_code == 200:
                    result = response.json()
                    predictions.append(result['fraud_prediction'])
                    probabilities.append(result['fraud_probability'])

                    # Send email alert if high risk
                    if user_email and result['fraud_probability'] > 0.9:
                        send_email_alert(row['title'], result['fraud_probability'], user_email)
                else:
                    raise Exception(f"Status Code {response.status_code}")

            except Exception as e:
                st.error(f"Prediction failed for row {index}: {e}")
                predictions.append(0)
                probabilities.append(0.0)

        df['fraud_prediction'] = predictions
        df['fraud_probability'] = probabilities

        st.success("✅ Predictions completed.")

        st.write("### Prediction Results")
        st.dataframe(df[['title', 'fraud_prediction', 'fraud_probability']])

        st.write("### Fraud Probability Distribution")
        fig, ax = plt.subplots()
        ax.hist(probabilities, bins=20, color='orange', edgecolor='black')
        ax.set_xlabel('Fraud Probability')
        ax.set_ylabel('Number of Postings')
        st.pyplot(fig)

        st.write("### Summary")
        st.write(f"Total postings: {len(df)}")
        st.write(f"Flagged as fraudulent: {df['fraud_prediction'].sum()} ({df['fraud_prediction'].mean()*100:.1f}%)")

        st.write("### 📊 Prediction Count (Legit vs Fraudulent)")
        count_data = df['fraud_prediction'].value_counts().rename({0: 'Legitimate', 1: 'Fraudulent'})
        st.bar_chart(count_data)

        st.write("### 🥧 Prediction Distribution (Pie Chart)")
        pie_labels = ['Legitimate', 'Fraudulent']
        pie_counts = [count_data.get('Legitimate', 0), count_data.get('Fraudulent', 0)]
        pie_colors = ['#00cc96', '#ff6361']
        pie_fig, pie_ax = plt.subplots()
        pie_ax.pie(pie_counts, labels=pie_labels, autopct='%1.1f%%', colors=pie_colors, startangle=90)
        pie_ax.axis('equal')
        st.pyplot(pie_fig)

        st.write("### 🔝 Top 10 Most Suspicious Job Postings")
        top10 = df.sort_values(by='fraud_probability', ascending=False).head(10)
        st.dataframe(top10[['title', 'company_profile', 'fraud_probability']])

        df['title_length'] = df['title'].apply(lambda x: len(str(x)))
        st.write("### 📈 Title Length vs Fraud Probability")
        scatter_fig, scatter_ax = plt.subplots()
        sns.scatterplot(x='title_length', y='fraud_probability', data=df, ax=scatter_ax, hue='fraud_prediction', palette='Set2')
        scatter_ax.set_xlabel("Job Title Length")
        scatter_ax.set_ylabel("Fraud Probability")
        scatter_ax.set_title("Title Length vs Fraud Probability")
        st.pyplot(scatter_fig)

        st.write("### 🧾 Recommendations")
        st.markdown("""
        - Postings with **very short or overly generic titles** may be suspicious.
        - Be cautious of listings with **missing company descriptions** or vague requirements.
        - **Avoid jobs that offer unrealistic benefits or salaries**.
        - If a job asks for personal details or upfront payments, **verify before responding**.
        """)

        st.download_button(
            label="📥 Download Results as CSV",
            data=df.to_csv(index=False),
            file_name='fraud_predictions.csv',
            mime='text/csv'
        )

with tab2:
    st.markdown("## 📊 About This App")
    st.write("""
    **Fake Job Posting Detector** is an intelligent tool that helps users detect potentially fraudulent job listings.

    💡 **How it works**:
    - Upload a CSV file containing job listings.
    - The app uses a trained Machine Learning model to analyze job text.
    - It flags job postings likely to be fake and shows the prediction probability.

    🔐 **Why this matters**:
    - Fake job postings are common on the internet.
    - This tool helps companies, job seekers, and platforms reduce fraud.

    🧠 **Technology stack**:
    - Python, Streamlit
    - FastAPI backend with Scikit-learn
    - Joblib for model loading
    - Google Drive integration for large file uploads
    - Seaborn & Matplotlib for charting

    👨‍💻 Built with ❤️ for social impact.
    """)
