#!/usr/bin/env python3

import os
import zipfile
import math
import time
import multiprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.simplefilter(action='ignore', category=pd.errors.ParserWarning)

# ---------------------- Haversine Distance ----------------------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = (math.sin(d_lat / 2))**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * (math.sin(d_lon / 2))**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

# ---------------------- GPS Spoofing Detection ----------------------
def detect_spoofing_for_vessel(args):
    mmsi_value, vessel_df = args
    if len(vessel_df) < 2:
        vessel_df["Distance_km"] = 0
        vessel_df["DeltaTime_hr"] = 0
        vessel_df["Speed_km_hr"] = 0
        vessel_df["Suspicious"] = False
        return vessel_df

    vessel_df = vessel_df.sort_values(by="Timestamp")
    vessel_df["lat_shift"] = vessel_df["Latitude"].shift(1)
    vessel_df["lon_shift"] = vessel_df["Longitude"].shift(1)
    vessel_df["time_shift"] = vessel_df["Timestamp"].shift(1)
    vessel_df.dropna(subset=["lat_shift", "lon_shift", "time_shift"], inplace=True)

    def compute_anomalies(row):
        dist = haversine(row["lat_shift"], row["lon_shift"], row["Latitude"], row["Longitude"])
        time_diff_hrs = (row["Timestamp"] - row["time_shift"]).total_seconds() / 3600.0
        speed_km_hr = dist / time_diff_hrs if time_diff_hrs > 0 else 0
        return pd.Series([dist, time_diff_hrs, speed_km_hr], index=["Distance_km", "DeltaTime_hr", "Speed_km_hr"])

    anomalies_df = vessel_df.apply(compute_anomalies, axis=1)
    vessel_df = pd.concat([vessel_df, anomalies_df], axis=1)

    SPEED_THRESHOLD = 200
    TIME_THRESHOLD_HR = 1/6.0
    DISTANCE_THRESHOLD = 100

    cond_speed = vessel_df["Speed_km_hr"] > SPEED_THRESHOLD
    cond_jump = (vessel_df["Distance_km"] > DISTANCE_THRESHOLD) & (vessel_df["DeltaTime_hr"] < TIME_THRESHOLD_HR)
    vessel_df["Suspicious"] = cond_speed | cond_jump
    return vessel_df

# ---------------------- Load and Clean Data ----------------------
def load_and_clean_data(zip_path, extract_path):
    print(f"Unzipping data from: {zip_path}")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(extract_path)
    print(f"Data extracted to: {extract_path}")

    csv_files = [f for f in os.listdir(extract_path) if f.lower().endswith('.csv')]
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {extract_path}")

    data_path = os.path.join(extract_path, csv_files[0])
    print(f"\nReading CSV file: {data_path}")
    df = pd.read_csv(data_path)

    print("Raw CSV columns:", df.columns.tolist())
    rename_map = {"# Timestamp": "Timestamp"}
    df.rename(columns=rename_map, inplace=True)
    print("Columns after rename:", df.columns.tolist())

    required_cols = ["MMSI", "Latitude", "Longitude", "Timestamp"]
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"Expected column '{col}' not found in DataFrame after renaming.")

    df.dropna(subset=required_cols, inplace=True)
    df = df[(df["Latitude"] >= -90) & (df["Latitude"] <= 90)]
    df = df[(df["Longitude"] >= -180) & (df["Longitude"] <= 180)]
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors='coerce')
    df.dropna(subset=["Timestamp"], inplace=True)
    df.sort_values(by=["MMSI", "Timestamp"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"Cleaned DataFrame shape: {df.shape}")
    return df

# ---------------------- Sequential vs Parallel ----------------------
def run_sequential(df):
    results = []
    for mmsi, group_df in df.groupby("MMSI"):
        out_df = detect_spoofing_for_vessel((mmsi, group_df.copy()))
        results.append(out_df)
    return pd.concat(results, ignore_index=True)

def run_parallel(df, num_processes=4):
    chunks = [(mmsi, group_df.copy()) for (mmsi, group_df) in df.groupby("MMSI")]
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.map(detect_spoofing_for_vessel, chunks)
    return pd.concat(results, ignore_index=True)

# ---------------------- Visualization ----------------------
def visualize_performance(df, sequential_time):
    cpu_counts = [2, 4, 6, 8]
    parallel_times = []

    print("\n📊 Testing performance with different CPU counts:")
    for cpu in cpu_counts:
        print(f"Running with {cpu} processes...")
        start = time.time()
        run_parallel(df, num_processes=cpu)
        end = time.time()
        elapsed = end - start
        parallel_times.append(elapsed)
        print(f"Time with {cpu} CPUs: {elapsed:.2f} seconds")

    plt.figure(figsize=(8, 5))
    plt.plot(cpu_counts, parallel_times, marker='o', label="Parallel Time")
    plt.axhline(y=sequential_time, color='red', linestyle='--', label="Sequential Time")
    plt.xlabel("Number of CPUs")
    plt.ylabel("Execution Time (seconds)")
    plt.title("Parallel Execution Time vs. CPU Count")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ---------------------- Main ----------------------
def main():
    zip_path = "/Users/emon/Documents/study file/2nd Semster/Big Data/Dataset/aisdk-2025-02-14.zip"
    extract_path = "/Users/emon/Documents/study file/2nd Semster/Big Data/Dataset/extracted_ais"
    result_path = "/Users/emon/Documents/study file/2nd Semster/Big Data/Dataset/spoofing_results.csv"
    summary_path = "/Users/emon/Documents/study file/2nd Semster/Big Data/Dataset/spoofing_summary.csv"

    df = load_and_clean_data(zip_path, extract_path)

    print("\nRunning spoofing detection: SEQUENTIAL")
    start_seq = time.time()
    df_seq = run_sequential(df)
    end_seq = time.time()
    seq_time = end_seq - start_seq
    print(f"Sequential run: {seq_time:.2f} seconds")

    print("\nRunning spoofing detection: PARALLEL with 4 processes")
    start_par = time.time()
    df_par = run_parallel(df, num_processes=4)
    end_par = time.time()
    par_time = end_par - start_par
    print(f"Parallel run: {par_time:.2f} seconds")

    speedup = seq_time / par_time if par_time else float('inf')
    print(f"\n🚀 Speedup = {speedup:.2f}x")

    df_par.to_csv(result_path, index=False)
    print(f"✅ Results saved to: {result_path}")
    print(f"🚨 Total suspicious events detected: {df_par['Suspicious'].sum()}")

    summary_df = df_par.groupby("MMSI")["Suspicious"].sum().reset_index()
    summary_df.to_csv(summary_path, index=False)
    print(f"📄 Spoofing summary saved to: {summary_path}")

    # Visualize performance
    visualize_performance(df, sequential_time=seq_time)

if __name__ == '__main__':
    main()
