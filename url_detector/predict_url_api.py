from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import re
import pandas as pd

app = FastAPI()

# ✅ Load the trained pipeline model
model = joblib.load("../models/hybrid_tfidf_model.joblib")

# ✅ Feature extractor for numeric features
def extract_numeric_features(url: str):
    # Normalize URL (remove scheme and www)
    normalized_url = re.sub(r'^(?:https?:\/\/)?(?:www\.)?', '', url.lower())

    url_length = len(url)
    domain = normalized_url.split('/')[0]
    domain_length = len(domain)
    num_digits = sum(c.isdigit() for c in url)
    num_dots = url.count('.')
    is_https = 1 if url.startswith("https") else 0
    suspicious_keywords = ['login', 'secure', 'account', 'update', 'verify', 'banking', 'signin']
    has_suspicious_keywords = 1 if any(keyword in url.lower() for keyword in suspicious_keywords) else 0

    return {
        'url': normalized_url,  # For TF-IDF
        'url_length': url_length,
        'domain_length': domain_length,
        'num_digits': num_digits,
        'num_dots': num_dots,
        'is_https': is_https,
        'has_suspicious_keywords': has_suspicious_keywords
    }

# ✅ Request schema
class URLRequest(BaseModel):
    url: str

# ✅ Response schema
class URLResponse(BaseModel):
    url: str
    prediction: str
    malicious_probability: float
    confidence: str

@app.post("/predict/", response_model=URLResponse)
async def predict_url(request: URLRequest):
    url = request.url
    features = extract_numeric_features(url)

    # Convert to DataFrame for the pipeline
    X = pd.DataFrame([features])

    # Predict
    prediction = model.predict(X)[0]
    proba = model.predict_proba(X)[0][1] * 100  # Probability of being malicious

    return {
        "url": url,
        "prediction": "malicious" if prediction == 1 else "benign",
        "malicious_probability": round(proba, 2),
        "confidence": f"{round(proba, 2)}%"
    }
