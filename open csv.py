import pandas as pd

file_path = "/Users/emon/Documents/study file/2nd Semster/Big Data/Dataset/spoofing_results.csv"
df = pd.read_csv(file_path)

# Check the first few rows
print(df.head())

# Count total flagged spoofing events
print("Total Suspicious Events:", df["Suspicious"].sum())
