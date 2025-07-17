import sys
import pandas as pd
import joblib
from feature_extractor import extract_features

def predict_url(input_url):
    # Load model
    model = joblib.load("../models/ensemble_tfidf_model.joblib")

    # Extract features
    features = extract_features(input_url)
    X = pd.DataFrame([features])

    # Predict
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0][1] * 100  # Probability of malicious

    result = {
        "url": input_url,
        "prediction": "malicious" if pred == 1 else "benign",
        "malicious_probability": round(proba, 2),
        "confidence": f"{round(proba, 2)}%"
    }

    print(result)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict_url.py <URL>")
    else:
        predict_url(sys.argv[1])
