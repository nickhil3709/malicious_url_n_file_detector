from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
from feature_extractor import extract_features

# Initialize FastAPI app
app = FastAPI(title="Malicious URL Detection API", description="Predict if a URL is malicious using a hybrid ML model.")

# Load the trained model
model = joblib.load("../models/ensemble_tfidf_model.joblib")

# Request schema
class URLItem(BaseModel):
    url: str

@app.get("/")
def root():
    return {"message": "Malicious URL Detection API is running!"}

@app.post("/predict/")
def predict_url(item: URLItem):
    # Extract features from the URL
    features = extract_features(item.url)

    # Convert to DataFrame for the model
    X = pd.DataFrame([features])

    # Make prediction and probability
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0][1] * 100  # Probability of malicious

    # Return response
    return {
        "url": item.url,
        "prediction": "malicious" if pred == 1 else "benign",
        "malicious_probability": round(proba, 2),
        "confidence": f"{round(proba, 2)}%"
    }

# To run:
# uvicorn predict_url_api:app --reload
