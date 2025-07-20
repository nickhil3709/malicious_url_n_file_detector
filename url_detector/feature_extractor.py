import re
import tldextract
import pandas as pd

# Suspicious keywords to check for
SUSPICIOUS_KEYWORDS = ['login', 'secure', 'update', 'verify', 'account', 'bank', 'paypal']

def extract_features(url: str):
    """Extracts numeric and text-based features from a URL."""
    extracted = tldextract.extract(url)
    domain = extracted.domain
    suffix = extracted.suffix

    features = {
        'url_length': len(url),
        'domain_length': len(domain),
        'num_digits': sum(c.isdigit() for c in url),
        'num_dots': url.count('.'),
        'is_https': 1 if url.startswith("https") else 0,
        'has_suspicious_keywords': 1 if any(word in url.lower() for word in SUSPICIOUS_KEYWORDS) else 0,
        # Keep original components for reference
        'domain': domain,
        'suffix': suffix,
        'url': url
    }
    return features

def build_feature_dataframe(df: pd.DataFrame):
    """Build feature DataFrame for all URLs in the dataset."""
    rows = []
    for _, row in df.iterrows():
        url = row['url']
        url_type = row['type']  # Multi-class: benign, defacement, phishing, malware
        features = extract_features(url)
        features['type'] = url_type  # Keep original class label
        rows.append(features)
    return pd.DataFrame(rows)

if __name__ == "__main__":
    input_file = "../data/malicious_phish_extended.csv"
    output_file = "../data/feature_data.csv"

    print(f"Loading dataset from {input_file}...")
    df_raw = pd.read_csv(input_file)

    print("Extracting features...")
    df_features = build_feature_dataframe(df_raw)

    print(f"Saving features to {output_file}...")
    df_features.to_csv(output_file, index=False)
    print("✅ Feature extraction complete!")
