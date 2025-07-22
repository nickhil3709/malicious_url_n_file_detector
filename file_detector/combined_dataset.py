import pandas as pd

benign = pd.read_csv("../data/1000_whitelist_sample_benign.csv")
malware = pd.read_csv("../data/1000_malware_sample_malicious.csv")

benign['CLASSIFICATION'] = 0
malware['CLASSIFICATION'] = 1

df = pd.concat([benign, malware], ignore_index=True)
df.to_csv("../data/combined_dataset.csv", index=False)


print("✅ Combined dataset saved as combined_dataset.csv")