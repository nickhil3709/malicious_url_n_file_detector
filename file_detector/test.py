import pandas as pd 

df = pd.read_csv("../data/PDFMalware2022.csv")
print(df.columns)
print("text values:")
print(df['text'].value_counts())