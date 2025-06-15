# predict_api.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib

# Load the trained model
model = joblib.load("fraud_job_model.pkl")

# Define the API app
app = FastAPI(
    title="Real-Time Job Scam Detector API",
    description="Detects whether a job posting is potentially fraudulent using a trained ML model.",
    version="1.0.0"
)

# CORS Middleware (important for external/public API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input schema using Pydantic
class JobPosting(BaseModel):
    title: str
    company_profile: str
    description: str
    requirements: str

# Health check
@app.get("/")
def read_root():
    return {
        "message": "✅ Scam Detection API is running. Use the /predict endpoint to POST job data."
    }

# Predict endpoint
@app.post("/predict")
def predict_job(job: JobPosting):
    try:
        # Combine fields
        combined_text = f"{job.title} {job.company_profile} {job.description} {job.requirements}"

        # Predict
        prediction = model.predict([combined_text])[0]
        probability = model.predict_proba([combined_text])[0][1]

        return {
            "fraud_prediction": int(prediction),
            "fraud_probability": float(probability)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
