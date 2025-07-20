import re
import os
import json
import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup

SUSPICIOUS_API_FILE = "../data/suspicious_api.json"


def fetch_suspicious_api_list(force_update=False):
    """Fetch suspicious Windows API names from malapi.io and cache locally."""
    if os.path.exists(SUSPICIOUS_API_FILE) and not force_update:
        with open(SUSPICIOUS_API_FILE, "r") as f:
            return json.load(f)
        
    print("[*] Fetching suspicious API names from malapi.io...")
    url = "https://malapi.io/"
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
    except Exception as e:
        raise Exception(f"Failed to fetch {url}, error: {e}")

    soup = BeautifulSoup(r.text, "html.parser")
    api_elements = soup.select("table tbody tr td:first-child a")
    api_names = [api.text.strip() for api in api_elements if api.text.strip()]

    print(f"[+] Found {len(api_names)} suspicious APIs.")
    os.makedirs(os.path.dirname(SUSPICIOUS_API_FILE), exist_ok=True)
    with open(SUSPICIOUS_API_FILE, "w") as f:
        json.dump(api_names, f, indent=2)

    return api_names


try:
    SUSPICIOUS_API_KEYWORDS = fetch_suspicious_api_list()
except Exception as e:
    print(f"[!] Warning: Could not fetch API list, using fallback. Error: {e}")
    SUSPICIOUS_API_KEYWORDS = ["VirtualAlloc", "WriteProcessMemory", "CreateRemoteThread", "LoadLibrary"]


def calculate_entropy(data: str) -> float:
    """
    Calculate Shannon entropy of the given string.
    Higher entropy often suggests packed or obfuscated code.
    """
    if not data or len(data) == 0:
        return 0.0
    data = data.encode('utf-8', errors='ignore')
    byte_counts = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256)
    probs = byte_counts / len(data)
    probs = probs[probs > 0]
    return round(-np.sum(probs * np.log2(probs)), 4)


def extract_file_features(row):
    features = {}

    # File size
    size = row.get('SIZE', 0)
    features['file_size'] = int(size) if pd.notna(size) and str(size).isdigit() else 0

    # Digital signature
    sig = row.get('DIGITAL SIG. ', 0)
    try:
        features['has_digital_signature'] = 1 if int(sig) == 1 else 0
    except:
        features['has_digital_signature'] = 0

    # Library features
    libraries = str(row.get('LIBRARIES', '')).split(',') if pd.notna(row.get('LIBRARIES')) else []
    libraries_clean = [lib.strip() for lib in libraries if lib.strip()]
    features['num_libraries'] = len(libraries_clean)
    features['lib_entropy'] = calculate_entropy(",".join(libraries_clean))

    # Function features
    functions = str(row.get('FUNCTIONS', '')).split(',') if pd.notna(row.get('FUNCTIONS')) else []
    functions_clean = [fn.strip() for fn in functions if fn.strip()]
    features['num_functions'] = len(functions_clean)
    features['func_entropy'] = calculate_entropy(",".join(functions_clean))

    # Suspicious API presence
    api_matches = [fn for fn in functions_clean if any(api.lower() in fn.lower() for api in SUSPICIOUS_API_KEYWORDS)]
    features['num_suspicious_apis'] = len(api_matches)

    # Class label
    features['label'] = int(row.get('CLASSIFICATION', 0))

    return features


def build_feature_dataframe(df):
    """
    Build enhanced feature dataframe from raw dataset.
    """
    rows = []
    for _, row in df.iterrows():
        rows.append(extract_file_features(row))

    return pd.DataFrame(rows)


if __name__ == "__main__":
    # Test with sample dataset
    df = pd.read_csv("../data/combined_dataset.csv")
    feature_df = build_feature_dataframe(df)
    feature_df.to_csv("../data/feature_pe_data.csv", index=False)
    print("[+] Feature extraction completed. Saved to ../data/feature_pe_data.csv")
    print(feature_df.head())
