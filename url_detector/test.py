import pandas as pd

df = pd.read_csv("../data/malicious_phish.csv")

print(df['type'].value_counts())