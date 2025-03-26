import pandas as pd
import matplotlib.pyplot as plt

# ✅ Load the results file
file_path = "/Users/emon/Documents/study file/2nd Semster/Big Data/Dataset/spoofing_results.csv"
df = pd.read_csv(file_path)

# ✅ Check if "Speed_km_hr" column exists
if "Speed_km_hr" not in df.columns:
    raise KeyError("Column 'Speed_km_hr' not found in dataset. Check your CSV file.")

# ✅ Plot histogram of vessel speeds
plt.hist(df["Speed_km_hr"], bins=50, color='blue', alpha=0.7)
plt.xlabel("Speed (km/h)")
plt.ylabel("Number of Records")
plt.title("Distribution of Vessel Speeds")
plt.show()
