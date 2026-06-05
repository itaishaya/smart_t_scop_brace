"""
T-Scope Section 8 Data Analysis Script (Upgraded ML Pipeline)
Author: Elia / Itai project support

Purpose:
1. Read calibration CSV files and infer sensor orientation.
2. Read measurement CSV files from two MPU6050 sensors.
3. Train Supervised ML Models using grouped 3-class datasets.
4. Predict activities on unseen mixed-activity data.
5. Export summary tables and plots for the project book.
"""

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# ============================================================
# 1. FILE PATHS
# ============================================================

CALIBRATION_FILES = {
    "standing": "cal_measurement_tscope_standing.csv",
    "horizontal_box_up": "cal_measurement_tscope_on_the_floor_box_upwards.csv",
    "lateral_side": "cal_measurement_tscope_on_the_floor_box_sideway.csv",
}

# Grouping the newly added files into the 3 classic categories. 
# (Excluded 'sitting_standing.csv' from pure training to avoid mixed-label contamination)
TRAINING_FILES = {
    "Walking": ["walking2.csv"],
    "Sitting": ["sitting_straightening_leg.csv"],
    "Standing": ["standing_bending.csv", "standing_raising_leg_no_bend.csv"]
}

# The files used for inference and general gait analysis
MEASUREMENT_FILES = {
    "walking": "walking.csv", # Used for ROM and peak detection
    "mixed_activity": "activities.csv", # Used as the unseen test set for ML
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "section8_outputs")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
TABLES_DIR = os.path.join(OUTPUT_DIR, "tables")

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def resolve_path(filename: str) -> str:
    candidates = [
        os.path.join(BASE_DIR, filename),
        os.path.join(os.getcwd(), filename),
        filename,
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"Could not find file: {filename}")


def load_csv(filename: str, default_fs=50.0) -> pd.DataFrame:
    path = resolve_path(filename)
    df = pd.read_csv(path)

    required_cols = [
        "sample", "time",
        "s1_ax", "s1_ay", "s1_az", "s1_gx", "s1_gy", "s1_gz",
        "s2_ax", "s2_ay", "s2_az", "s2_gx", "s2_gy", "s2_gz",
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {filename}: {missing}")

    df = df.copy()
    df["time_sec"] = df["time"] - df["time"].iloc[0]
    df["dt"] = df["time_sec"].diff()
    median_dt = df["dt"].median()

    # SAFEGUARD: Fix for identical timestamps (integer seconds) or missing data
    if pd.isna(median_dt) or median_dt <= 0:
        total_time = df["time_sec"].iloc[-1]
        if total_time > 0:
            median_dt = total_time / len(df)
        else:
            median_dt = 1.0 / default_fs # Fallback to 50 Hz (0.02s)

    df["dt"] = df["dt"].fillna(median_dt)
    df.loc[df["dt"] <= 0, "dt"] = median_dt
    return df


def estimate_sampling_rate(df: pd.DataFrame, default_fs=50.0) -> float:
    median_dt = df["dt"].median()
    if pd.isna(median_dt) or median_dt <= 0:
        return default_fs
    return 1.0 / median_dt


def moving_average(x, window=7):
    return pd.Series(x).rolling(window=window, center=True, min_periods=1).mean().to_numpy()


def simple_peak_detection(signal, min_distance_samples=50, threshold=None):
    signal = np.asarray(signal)
    if threshold is None:
        threshold = np.mean(signal) + 0.8 * np.std(signal)

    candidate_indices = []
    for i in range(1, len(signal) - 1):
        if signal[i] > signal[i - 1] and signal[i] > signal[i + 1] and signal[i] > threshold:
            candidate_indices.append(i)

    if not candidate_indices:
        return np.array([], dtype=int)

    selected = []
    selected.append(candidate_indices[0])

    for idx in candidate_indices[1:]:
        if idx - selected[-1] >= min_distance_samples:
            selected.append(idx)
        else:
            if signal[idx] > signal[selected[-1]]:
                selected[-1] = idx
    return np.array(selected, dtype=int)


def save_plot(filename):
    full_path = os.path.join(PLOTS_DIR, filename)
    plt.tight_layout()
    plt.savefig(full_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {full_path}")


# ============================================================
# 3. CALIBRATION & KNEE ANGLE ESTIMATION
# ============================================================

def analyze_calibration():
    rows = []
    for cal_name, file_name in CALIBRATION_FILES.items():
        df = load_csv(file_name)
        row = {
            "calibration_position": cal_name,
            "samples": len(df),
            "sampling_rate_hz": estimate_sampling_rate(df),
        }
        for col in ["s1_ax", "s1_ay", "s1_az", "s2_ax", "s2_ay", "s2_az"]:
            row[col + "_mean_g"] = df[col].mean()
            row[col + "_std_g"] = df[col].std()
        rows.append(row)

    cal_summary = pd.DataFrame(rows)
    cal_summary.to_csv(os.path.join(TABLES_DIR, "calibration_summary.csv"), index=False)
    return cal_summary


def add_anatomical_axes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["lower_vertical_g"] = -df["s1_ax"]
    df["lower_anterior_g"] = df["s1_az"]
    df["upper_vertical_g"] = -df["s2_ay"]
    df["upper_anterior_g"] = df["s2_az"]
    df["lower_flexion_rate_dps"] = df["s1_gy"]
    df["upper_flexion_rate_dps"] = -df["s2_gx"]
    df["lower_acc_angle_deg"] = np.degrees(np.arctan2(df["lower_anterior_g"], df["lower_vertical_g"]))
    df["upper_acc_angle_deg"] = np.degrees(np.arctan2(df["upper_anterior_g"], df["upper_vertical_g"]))
    return df


def complementary_filter(acc_angle_deg, gyro_rate_dps, dt, alpha=0.98):
    acc_angle_deg = np.asarray(acc_angle_deg)
    gyro_rate_dps = np.asarray(gyro_rate_dps)
    dt = np.asarray(dt)
    angle = np.zeros(len(acc_angle_deg))
    angle[0] = acc_angle_deg[0]
    for i in range(1, len(acc_angle_deg)):
        gyro_prediction = angle[i - 1] + gyro_rate_dps[i] * dt[i]
        angle[i] = alpha * gyro_prediction + (1 - alpha) * acc_angle_deg[i]
    return angle


def add_knee_angle(df: pd.DataFrame, alpha=0.98) -> pd.DataFrame:
    df = add_anatomical_axes(df)
    df["lower_angle_deg"] = complementary_filter(df["lower_acc_angle_deg"], df["lower_flexion_rate_dps"], df["dt"], alpha=alpha)
    df["upper_angle_deg"] = complementary_filter(df["upper_acc_angle_deg"], df["upper_flexion_rate_dps"], df["dt"], alpha=alpha)
    df["knee_angle_raw_deg"] = df["lower_angle_deg"] - df["upper_angle_deg"]
    first_second = df["time_sec"] <= min(1.0, df["time_sec"].max())
    reference = df.loc[first_second, "knee_angle_raw_deg"].median()
    df["knee_angle_deg"] = df["knee_angle_raw_deg"] - reference
    df["knee_angle_smooth_deg"] = moving_average(df["knee_angle_deg"], window=15)
    return df


def add_motion_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["lower_acc_norm_g"] = np.sqrt(df["s1_ax"]**2 + df["s1_ay"]**2 + df["s1_az"]**2)
    df["upper_acc_norm_g"] = np.sqrt(df["s2_ax"]**2 + df["s2_ay"]**2 + df["s2_az"]**2)
    df["lower_gyro_norm_dps"] = np.sqrt(df["s1_gx"]**2 + df["s1_gy"]**2 + df["s1_gz"]**2)
    df["upper_gyro_norm_dps"] = np.sqrt(df["s2_gx"]**2 + df["s2_gy"]**2 + df["s2_gz"]**2)
    
    df["motion_intensity"] = (
        df["lower_gyro_norm_dps"] + df["upper_gyro_norm_dps"] +
        40 * np.abs(df["lower_acc_norm_g"] - 1.0) +
        40 * np.abs(df["upper_acc_norm_g"] - 1.0)
    )
    df["motion_intensity_smooth"] = moving_average(df["motion_intensity"], window=25)
    
    df["lower_acc_norm_smooth"] = moving_average(df["lower_acc_norm_g"], window=7)
    df["jerk"] = np.gradient(df["lower_acc_norm_smooth"], df["time_sec"])
    df["abs_jerk_smooth"] = moving_average(np.abs(df["jerk"]), window=15)
    return df


def plot_raw_walking_signals(df: pd.DataFrame):
    plt.figure(figsize=(12, 5))
    plt.plot(df["time_sec"], df["s1_ax"], label="s1_ax")
    plt.plot(df["time_sec"], df["s1_ay"], label="s1_ay")
    plt.plot(df["time_sec"], df["s1_az"], label="s1_az")
    plt.title("Lower Sensor (0x68) Accelerometer During Walking")
    plt.xlabel("Time [s]")
    plt.ylabel("Acceleration [g]")
    plt.grid(True)
    plt.legend()
    save_plot("02_lower_sensor_accelerometer_walking.png")

    plt.figure(figsize=(12, 5))
    plt.plot(df["time_sec"], df["s2_ax"], label="s2_ax")
    plt.plot(df["time_sec"], df["s2_ay"], label="s2_ay")
    plt.plot(df["time_sec"], df["s2_az"], label="s2_az")
    plt.title("Upper Sensor (0x69) Accelerometer During Walking")
    plt.xlabel("Time [s]")
    plt.ylabel("Acceleration [g]")
    plt.grid(True)
    plt.legend()
    save_plot("03_upper_sensor_accelerometer_walking.png")

    plt.figure(figsize=(12, 5))
    plt.plot(df["time_sec"], df["s1_gx"], label="s1_gx")
    plt.plot(df["time_sec"], df["s1_gy"], label="s1_gy")
    plt.plot(df["time_sec"], df["s1_gz"], label="s1_gz")
    plt.title("Lower Sensor (0x68) Gyroscope During Walking")
    plt.xlabel("Time [s]")
    plt.ylabel("Angular Velocity [deg/s]")
    plt.grid(True)
    plt.legend()
    save_plot("04_lower_sensor_gyroscope_walking.png")

    plt.figure(figsize=(12, 5))
    plt.plot(df["time_sec"], df["s2_gx"], label="s2_gx")
    plt.plot(df["time_sec"], df["s2_gy"], label="s2_gy")
    plt.plot(df["time_sec"], df["s2_gz"], label="s2_gz")
    plt.title("Upper Sensor (0x69) Gyroscope During Walking")
    plt.xlabel("Time [s]")
    plt.ylabel("Angular Velocity [deg/s]")
    plt.grid(True)
    plt.legend()
    save_plot("05_upper_sensor_gyroscope_walking.png")

def plot_knee_angle(df: pd.DataFrame, filename_prefix="walking"):
    plt.figure(figsize=(12, 5))
    plt.plot(df["time_sec"], df["knee_angle_deg"], label="Raw estimated knee angle", alpha=0.5)
    plt.plot(df["time_sec"], df["knee_angle_smooth_deg"], label="Smoothed knee angle", linewidth=2)
    plt.title("Estimated Knee Flexion-Extension Angle")
    plt.xlabel("Time [s]")
    plt.ylabel("Knee Angle [deg]")
    plt.grid(True)
    plt.legend()
    save_plot(f"06_{filename_prefix}_knee_angle_estimation.png")


# ============================================================
# 4. WALKING DETECTION AND ROM EXTRACTION
# ============================================================

def detect_steps_and_rom(df: pd.DataFrame):
    df = add_motion_features(df)
    fs = estimate_sampling_rate(df)
    min_step_distance_sec = 0.35
    min_distance_samples = max(5, int(min_step_distance_sec * fs))
    threshold = df["abs_jerk_smooth"].mean() + 0.9 * df["abs_jerk_smooth"].std()

    peaks = simple_peak_detection(df["abs_jerk_smooth"].to_numpy(), min_distance_samples=min_distance_samples, threshold=threshold)

    rows = []
    for i in range(len(peaks) - 1):
        start_idx = peaks[i]
        end_idx = peaks[i + 1]
        segment = df.iloc[start_idx:end_idx + 1]
        max_flexion = segment["knee_angle_smooth_deg"].max()
        min_flexion = segment["knee_angle_smooth_deg"].min()
        rows.append({
            "stride_number": i + 1,
            "start_time_sec": df["time_sec"].iloc[start_idx],
            "end_time_sec": df["time_sec"].iloc[end_idx],
            "duration_sec": df["time_sec"].iloc[end_idx] - df["time_sec"].iloc[start_idx],
            "min_angle_deg": min_flexion,
            "max_angle_deg": max_flexion,
            "rom_deg": max_flexion - min_flexion,
        })

    rom_table = pd.DataFrame(rows)
    
    # Plot jerk peaks
    plt.figure(figsize=(12, 5))
    plt.plot(df["time_sec"], df["abs_jerk_smooth"], label="Smoothed absolute jerk")
    if len(peaks) > 0:
        plt.scatter(df["time_sec"].iloc[peaks], df["abs_jerk_smooth"].iloc[peaks], label="Detected gait events")
    plt.title("Walking Detection Using Jerk Peak Detection")
    plt.xlabel("Time [s]")
    plt.ylabel("Absolute Jerk [g/s]")
    plt.grid(True)
    plt.legend()
    save_plot("07_walking_detection_jerk_peaks.png")

    # Plot knee angle with detected steps
    plt.figure(figsize=(12, 5))
    plt.plot(df["time_sec"], df["knee_angle_smooth_deg"], label="Knee angle")
    if len(peaks) > 0:
        plt.scatter(df["time_sec"].iloc[peaks], df["knee_angle_smooth_deg"].iloc[peaks], label="Detected gait events")
    plt.title("Knee Angle with Detected Walking Events")
    plt.xlabel("Time [s]")
    plt.ylabel("Knee Angle [deg]")
    plt.grid(True)
    plt.legend()
    save_plot("08_knee_angle_with_detected_gait_events.png")

    # Plot ROM per stride
    if not rom_table.empty:
        plt.figure(figsize=(12, 5))
        plt.plot(rom_table["stride_number"], rom_table["rom_deg"], marker="o")
        plt.title("ROM per Detected Walking Stride")
        plt.xlabel("Stride Number")
        plt.ylabel("ROM [deg]")
        plt.grid(True)
        save_plot("09_rom_per_stride.png")
        
    return df, peaks, rom_table


# ============================================================
# 5. SUPERVISED MACHINE LEARNING PIPELINE
# ============================================================

def train_supervised_models():
    """
    Reads grouped training CSV files, extracts features, and trains the AI.
    """
    print("\n--- Training Supervised ML Models on Labeled Datasets ---")
    dfs = []
    
    for label, files in TRAINING_FILES.items():
        for filename in files:
            try:
                print(f"Loading '{label}' dataset: {filename}")
                df = load_csv(filename)
                df = add_knee_angle(df)
                df = add_motion_features(df)
                
                fs = estimate_sampling_rate(df)
                if math.isnan(fs) or fs <= 0:
                    fs = 50.0 
                    
                window_ml = max(1, int(1.0 * fs)) # 1-second rolling window
                
                # Extract ML Features
                df['angle_mean'] = df['knee_angle_smooth_deg'].rolling(window_ml, center=True).mean().bfill().ffill()
                df['angle_std'] = df['knee_angle_smooth_deg'].rolling(window_ml, center=True).std().fillna(0)
                df['jerk_mean'] = df['abs_jerk_smooth'].rolling(window_ml, center=True).mean().fillna(0)
                df['Label'] = label # Assign explicit 3-class ground truth
                
                dfs.append(df)
            except FileNotFoundError:
                print(f"WARNING: Training file {filename} not found. Skipping.")
            
    if not dfs:
        raise ValueError("No training files were found. Please check paths.")
        
    full_train_df = pd.concat(dfs, ignore_index=True)
    
    features_list = ['angle_mean', 'angle_std', 'jerk_mean', 's1_ax', 's2_ax']
    X = full_train_df[features_list]
    
    # Encode labels (Standing, Sitting, Walking)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(full_train_df['Label'])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    
    print("\nTraining Random Forest Classifier...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    print("Training XGBoost Classifier...")
    xgb_model = XGBClassifier(eval_metric='mlogloss', random_state=42)
    xgb_model.fit(X_train, y_train)
    
    print(f"\nRandom Forest Validation Accuracy: {accuracy_score(y_test, rf_model.predict(X_test)):.4f}")
    print(f"XGBoost Validation Accuracy: {accuracy_score(y_test, xgb_model.predict(X_test)):.4f}")
    
    return rf_model, xgb_model, label_encoder, features_list


def predict_activities(df, rf_model, xgb_model, label_encoder, features_list):
    """
    Applies the trained models to unseen mixed activity data.
    """
    df = add_knee_angle(df)
    df = add_motion_features(df)
    
    fs = estimate_sampling_rate(df)
    if math.isnan(fs) or fs <= 0:
        fs = 50.0
        
    window_ml = max(1, int(1.0 * fs))
    
    df['angle_mean'] = df['knee_angle_smooth_deg'].rolling(window_ml, center=True).mean().bfill().ffill()
    df['angle_std'] = df['knee_angle_smooth_deg'].rolling(window_ml, center=True).std().fillna(0)
    df['jerk_mean'] = df['abs_jerk_smooth'].rolling(window_ml, center=True).mean().fillna(0)
    
    X = df[features_list]
    
    df['RF_Activity'] = label_encoder.inverse_transform(rf_model.predict(X))
    df['XGB_Activity'] = label_encoder.inverse_transform(xgb_model.predict(X))
    
    return df


def plot_predicted_activities(df, prediction_col, model_name, filename_prefix="mixed_activity"):
    """
    Restored to use the classic 3 colors for Standing, Sitting, and Walking
    """
    plt.figure(figsize=(16, 6))
    plt.plot(df['time_sec'], df['knee_angle_smooth_deg'], label='Smoothed Knee Angle', color='black', alpha=0.5, linewidth=1.5)

    # Restored Old Color Mapping
    colors = {'Standing': 'orange', 'Sitting': 'blue', 'Walking': 'green'}

    for label, color in colors.items():
        act_mask = df[prediction_col] == label
        if act_mask.any(): 
            plt.scatter(df['time_sec'][act_mask], df['knee_angle_smooth_deg'][act_mask], 
                        color=color, label=label, s=12, alpha=0.7)

    plt.title(f"{model_name} Predicted Activity States and Knee ROM")
    plt.xlabel("Time [s]")
    plt.ylabel("Knee Flexion [deg]")

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper right')
    plt.grid(True)
    
    safe_model_name = model_name.lower().replace(" ", "_")
    save_plot(f"13_{filename_prefix}_{safe_model_name}_classification.png")


# ============================================================
# 6. MAIN PIPELINE
# ============================================================

def main():
    print("Starting T-Scope Upgraded ML Analysis Pipeline...")
    cal_summary = analyze_calibration()

    # 1. Walking Measurement (ROM analysis)
    print("\n--- Running Walking/Gait Analysis ---")
    walking_df = load_csv(MEASUREMENT_FILES["walking"])
    print(f"Estimated sampling rate: {estimate_sampling_rate(walking_df):.2f} Hz")
    
    walking_df = add_knee_angle(walking_df)
    plot_raw_walking_signals(walking_df)
    plot_knee_angle(walking_df, filename_prefix="walking")
    
    walking_df, peaks, rom_table = detect_steps_and_rom(walking_df)
    walking_df.to_csv(os.path.join(TABLES_DIR, "walking_processed_data.csv"), index=False)
    
    if not rom_table.empty:
        rom_table.to_csv(os.path.join(TABLES_DIR, "walking_stride_rom_table.csv"), index=False)
        print("ROM statistics extracted successfully.")

    # 2. Train the Models using Grouped Files
    rf_model, xgb_model, label_encoder, features_list = train_supervised_models()

    # 3. Predict on Mixed/Unseen Data
    print("\n--- Running Inference on Mixed Activity Data ---")
    mixed_df = load_csv(MEASUREMENT_FILES["mixed_activity"])
    mixed_df = predict_activities(mixed_df, rf_model, xgb_model, label_encoder, features_list)
    mixed_df.to_csv(os.path.join(TABLES_DIR, "mixed_activity_predictions.csv"), index=False)

    # Plot mixed activity original motion intensity / angle (from original script)
    plt.figure(figsize=(12, 5))
    plt.plot(mixed_df["time_sec"], mixed_df["motion_intensity_smooth"], label="Motion intensity")
    plt.title("Mixed Activity Measurement: Motion Intensity Over Time")
    plt.xlabel("Time [s]")
    plt.ylabel("Motion Intensity [a.u.]")
    plt.grid(True)
    plt.legend()
    save_plot("10_mixed_activity_motion_intensity.png")

    plt.figure(figsize=(12, 5))
    plt.plot(mixed_df["time_sec"], mixed_df["knee_angle_smooth_deg"], label="Knee angle")
    plt.title("Mixed Activity Measurement: Estimated Knee Angle")
    plt.xlabel("Time [s]")
    plt.ylabel("Knee Angle [deg]")
    plt.grid(True)
    plt.legend()
    save_plot("11_mixed_activity_knee_angle.png")

    # 4. Plot the ML classification results
    plot_predicted_activities(mixed_df, prediction_col='RF_Activity', model_name='Random Forest')
    plot_predicted_activities(mixed_df, prediction_col='XGB_Activity', model_name='XGBoost')

    print("\nDone. All outputs and plots were saved to:", OUTPUT_DIR)

if __name__ == "__main__":
    main()