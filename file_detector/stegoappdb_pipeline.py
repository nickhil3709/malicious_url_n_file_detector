import os
import requests
import zipfile
from tqdm import tqdm
from PIL import Image
import numpy as np
import pandas as pd

STEGO_URL = "http://agents.fel.cvut.cz/stegodata/StegoAppDB.zip"
DATA_DIR = "../data/stegoappdb"
RAW_DIR = os.path.join(DATA_DIR, "raw")
EXTRACT_DIR = os.path.join(DATA_DIR, "images")
FEATURES_CSV = os.path.join(DATA_DIR, "stego_features.csv")


def download_dataset():
    os.makedirs(DATA_DIR, exist_ok=True)
    zip_path = os.path.join(DATA_DIR, "StegoAppDB.zip")
    if os.path.exists(zip_path):
        print("[INFO] Dataset already downloaded.")
        return
    print("[INFO] Downloading StegoAppDB dataset...")
    response = requests.get(STEGO_URL, stream=True)
    total = int(response.headers.get("content-length", 0))
    with open(zip_path, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as pbar:
        for chunk in response.iter_content(1024 * 1024):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))
    print("[INFO] Download complete.")


def extract_dataset():
    zip_path = os.path.join(DATA_DIR, "StegoAppDB.zip")
    if not os.path.exists(zip_path):
        raise FileNotFoundError("Dataset ZIP not found. Run download_dataset() first.")
    if os.path.exists(EXTRACT_DIR):
        print("[INFO] Already extracted.")
        return
    print("[INFO] Extracting dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(EXTRACT_DIR)
    print("[INFO] Extraction complete.")


def calculate_features(image_path):
    img = Image.open(image_path).convert("RGB")
    arr = np.array(img)
    mean = arr.mean()
    std = arr.std()
    entropy = -np.sum(np.bincount(arr.flatten(), minlength=256) / arr.size *
                      np.log2(np.bincount(arr.flatten(), minlength=256) / arr.size + 1e-7))
    return mean, std, entropy


def build_feature_csv():
    if not os.path.exists(EXTRACT_DIR):
        raise FileNotFoundError("Images not found. Run extract_dataset() first.")
    records = []
    for root, _, files in os.walk(EXTRACT_DIR):
        for file in files:
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                path = os.path.join(root, file)
                mean, std, entropy = calculate_features(path)
                label = 1 if "stego" in root.lower() else 0
                records.append({
                    "filename": file,
                    "mean": round(mean, 3),
                    "std": round(std, 3),
                    "entropy": round(entropy, 3),
                    "label": label
                })
    df = pd.DataFrame(records)
    df.to_csv(FEATURES_CSV, index=False)
    print(f"[INFO] Features saved at {FEATURES_CSV}")


if __name__ == "__main__":
    download_dataset()
    extract_dataset()
    build_feature_csv()
