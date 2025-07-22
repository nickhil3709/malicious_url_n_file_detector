import os
import cv2
import numpy as np
import pandas as pd
from skimage.measure import shannon_entropy
from tqdm import tqdm

# -------------------------------
# CONFIG
# -------------------------------
DATASET_DIR = "../data/stegoappdb"  # Updated base directory
OUTPUT_DIR = "../data/csv_features"
SETS = ["train/train", "test/test", "val/val"] # data\stegoappdb\test\test\clean
# Image extensions
IMG_EXT = {".png", ".jpg", ".jpeg"}

# -------------------------------
# Feature Extraction
# -------------------------------
def extract_features(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)

    mean_r, mean_g, mean_b = np.mean(img[:, :, 0]), np.mean(img[:, :, 1]), np.mean(img[:, :, 2])
    std_r, std_g, std_b = np.std(img[:, :, 0]), np.std(img[:, :, 1]), np.std(img[:, :, 2])

    gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    entropy = shannon_entropy(gray)

    lsb = np.bitwise_and(gray.astype(np.uint8), 1)
    lsb_ratio = np.mean(lsb)

    return [mean_r, mean_g, mean_b, std_r, std_g, std_b, entropy, lsb_ratio]

def process_folder(folder_path, label):
    rows = []
    if not os.path.exists(folder_path):
        print(f"[WARNING] Folder not found: {folder_path}")
        return rows

    for fname in tqdm(os.listdir(folder_path), desc=f"Processing {folder_path}"):
        ext = os.path.splitext(fname)[1].lower()
        if ext not in IMG_EXT:
            continue
        fpath = os.path.join(folder_path, fname)
        features = extract_features(fpath)
        if features:
            rows.append(features + [label])
    return rows

# -------------------------------
# Main Pipeline
# -------------------------------
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    columns = [
        "mean_r", "mean_g", "mean_b",
        "std_r", "std_g", "std_b",
        "entropy", "lsb_ratio",
        "label"
    ]

    for subset in SETS:
        clean_dir = os.path.join(DATASET_DIR, subset, "clean")
        stego_dir = os.path.join(DATASET_DIR, subset, "stego")

        print(f"\n[INFO] Processing {subset}...")
        clean_rows = process_folder(clean_dir, 0)
        stego_rows = process_folder(stego_dir, 1)

        all_rows = clean_rows + stego_rows
        if all_rows:
            df = pd.DataFrame(all_rows, columns=columns)
            out_path = os.path.join(OUTPUT_DIR, f"{subset.replace('/', '_')}.csv")
            df.to_csv(out_path, index=False)
            print(f"[SAVED] {out_path} with {len(df)} samples.")
        else:
            print(f"[WARNING] No data found for {subset}.")
