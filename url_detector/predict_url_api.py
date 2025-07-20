from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import re
from typing import Dict

app = FastAPI(title="Malicious URL Multi-Class Detector")

# ✅ Load the trained multi-class model
model = joblib.load("../models/multi_class_tfidf_rf.joblib")

# ✅ Feature extractor
def extract_numeric_features(url: str):
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
        'url': normalized_url,
        'url_length': url_length,
        'domain_length': domain_length,
        'num_digits': num_digits,
        'num_dots': num_dots,
        'is_https': is_https,
        'has_suspicious_keywords': has_suspicious_keywords
    }

# ✅ Request Schema
class URLRequest(BaseModel):
    url: str

# ✅ Response Schema
class URLResponse(BaseModel):
    url: str
    predicted_class: str
    confidence: float
    class_probabilities: Dict[str, float]

@app.post("/predict/", response_model=URLResponse)
async def predict_url(request: URLRequest):
    url = request.url
    features = extract_numeric_features(url)

    X = pd.DataFrame([features])

    prediction = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]

    # Map probabilities to class names
    class_labels = model.classes_
    class_probs = {class_labels[i]: round(probabilities[i] * 100, 2) for i in range(len(class_labels))}

    return {
        "url": url,
        "predicted_class": prediction,
        "confidence": round(max(probabilities) * 100, 2),
        "class_probabilities": class_probs
    }
