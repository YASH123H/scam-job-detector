import smtplib
from email.mime.text import MIMEText
import os
from dotenv import load_dotenv

# Load sender credentials from .env file
load_dotenv()
EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASS = os.getenv("EMAIL_PASS")

def send_email_alert(job_title, prob, recipient_email):
    """Send fraud alert email to the user who uploaded the CSV."""
    body = f"""⚠️ Fraud Alert!

Job Title: '{job_title}'
Fraud Probability: {prob:.2f}

This job listing appears highly suspicious. Please exercise caution before applying.

- Automated Alert System
"""
    msg = MIMEText(body)
    msg['Subject'] = "🚨 Fake Job Detected"
    msg['From'] = EMAIL_USER
    msg['To'] = recipient_email

    try:
        server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        server.login(EMAIL_USER, EMAIL_PASS)
        server.send_message(msg)
        server.quit()
        print(f"✅ Email sent to {recipient_email} for job: {job_title}")
    except Exception as e:
        print(f"❌ Email failed for job: {job_title} → {e}")
