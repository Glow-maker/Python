import pandas as pd

# Load the Parquet file
df = pd.read_parquet(r'C:\Users\jintao\OneDrive\work\Vscode\sklearn\Regression\data\test1.parquet')

# Display the first few rows of the DataFrame
print(df.head())