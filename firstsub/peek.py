import pandas as pd
df = pd.read_csv(r'carbon_24/test.csv')
print("Columns:", df.columns.tolist())
print("Shape:", df.shape)
print(df.head(2).to_string())
