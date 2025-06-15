from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import joblib

# Load the trained model
model = joblib.load("fraud_job_model.pkl")

# Define the API app
app = FastAPI(
    title="Real-Time Job Scam Detector API",
    description="Detects whether a job posting is potentially fraudulent using a trained ML model.",
    version="1.0.0"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all for development; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input schema
class JobPosting(BaseModel):
    title: str
    company_profile: str
    description: str
    requirements: str

# Root endpoint
@app.get("/")
def read_root():
    return {
        "message": "✅ Scam Detection API is running. Use /predict for single and /predict_batch for multiple."
    }

# Single prediction endpoint
@app.post("/predict")
def predict_job(job: JobPosting):
    try:
        combined_text = f"{job.title} {job.company_profile} {job.description} {job.requirements}"
        prediction = model.predict([combined_text])[0]
        probability = model.predict_proba([combined_text])[0][1]

        return {
            "fraud_prediction": int(prediction),
            "fraud_probability": float(probability)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# ✅ Batch prediction endpoint
@app.post("/predict_batch")
def predict_batch(jobs: List[JobPosting]):
    try:
        # Prepare combined inputs
        combined_texts = [
            f"{job.title} {job.company_profile} {job.description} {job.requirements}"
            for job in jobs
        ]

        # Batch predictions
        predictions = model.predict(combined_texts)
        probabilities = model.predict_proba(combined_texts)

        results = []
        for pred, prob in zip(predictions, probabilities):
            results.append({
                "fraud_prediction": int(pred),
                "fraud_probability": float(prob[1])
            })

        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")
