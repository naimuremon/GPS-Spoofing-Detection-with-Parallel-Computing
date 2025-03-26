import pandas as pd

# Path to your summary file
file_path = "/Users/emon/Documents/study file/2nd Semster/Big Data/Dataset/spoofing_summary.csv"

# Load the summary
summary_df = pd.read_csv(file_path)

# Show first 10 rows
print("📄 Spoofing Summary (Top 10 Vessels):")
print(summary_df.head(10))

# Total vessels with at least 1 suspicious event
nonzero = summary_df[summary_df["Suspicious"] > 0].shape[0]
print(f"\n🚨 Vessels with spoofing activity: {nonzero}")
