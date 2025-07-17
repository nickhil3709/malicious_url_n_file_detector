import re
import tldextract
import pandas as pd
from urllib.parse import urlparse

import tldextract
import re

# List of suspicious keywords
SUSPICIOUS_KEYWORDS = ['login', 'secure', 'update', 'verify', 'account', 'bank', 'paypal']

def extract_features(url):
    extracted = tldextract.extract(url)
    domain = extracted.domain
    suffix = extracted.suffix

    features = {}
    features['url_length'] = len(url)
    features['domain_length'] = len(domain)
    features['num_digits'] = sum(c.isdigit() for c in url)
    features['num_dots'] = url.count('.')
    features['is_https'] = 1 if url.startswith("https") else 0
    features['has_suspicious_keywords'] = 1 if any(word in url.lower() for word in SUSPICIOUS_KEYWORDS) else 0

    # Keep these for reference (not for training)
    features['domain'] = domain
    features['suffix'] = suffix
    features['url'] = url

    return features

def build_feature_dataframe(df):
    rows = []
    for _,row in df.iterrows():
        url = row['url']
        type =1 if row['type'] == 'phishing' else 0
        features = extract_features(url)
        features['type'] = type
        rows.append(features)
    
    return pd.DataFrame(rows)

df = build_feature_dataframe(pd.read_csv("../data/malicious_phish.csv")) 
df.to_csv("../data/feature_data.csv", index=False)