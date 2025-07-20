import requests
import zipfile
import io
import pandas as pd

# ✅ Tranco Top 1M List (Updated alternative to Alexa)
TRANCO_URL = "https://tranco-list.eu/top-1m.csv.zip"

# Paths
DATASET_PATH = "../data/malicious_phish.csv"
OUTPUT_PATH = "../data/malicious_phish_extended.csv"

print("Downloading Tranco Top 1M ZIP file...")
response = requests.get(TRANCO_URL, stream=True)

if response.status_code == 200:
    print("Download successful. Extracting...")
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        # Extract the CSV file from the zip
        with z.open(z.namelist()[0]) as f:
            # Read top 1K domains into DataFrame
            tranco_df = pd.read_csv(f, header=None, names=["rank", "domain"], nrows=1000)
else:
    raise Exception(f"Failed to download Tranco list. Status code: {response.status_code}")

# ✅ Generate benign URLs (5 variations per domain)
domains = tranco_df["domain"].tolist()
benign_urls = []
for domain in domains:
    benign_urls.extend([
        f"http://{domain}",
        f"https://{domain}",
        f"https://www.{domain}",
        f"http://www.{domain}/index.html",
        f"https://{domain}/about"
    ])

# ✅ Create benign DataFrame
benign_df = pd.DataFrame({
    "url": benign_urls,
    "type": "benign"
})

# ✅ Load original dataset and append new benign URLs
original_df = pd.read_csv(DATASET_PATH)
extended_df = pd.concat([original_df, benign_df], ignore_index=True)

print(f"Original dataset size: {len(original_df)}")
print(f"Extended dataset size: {len(extended_df)}")

# ✅ Save the extended dataset
extended_df.to_csv(OUTPUT_PATH, index=False)
print(f"✅ Extended dataset saved to {OUTPUT_PATH}")
