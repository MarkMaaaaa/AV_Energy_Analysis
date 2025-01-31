
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.signal import welch

# Load both datasets
acc_data = pd.read_csv('ACC_processed_output.csv')
hv_data = pd.read_csv('HV_processed_output.csv')

# Define parameters for moving average and plot aesthetics
window_size = 10  # Moving average window size
font_size = 22  # Font size for labels and titles
line_width = 4  # Line width for plots
sample_rate = 1
# Ensure both datasets have the same trajectories
unique_trajectories = acc_data['trajectory_id'].unique()

results_list = []

# Loop through each trajectory, apply moving average, and plot residuals
plt.figure(figsize=(12, 8))

# Loop through each trajectory and calculate PSD for ACC and HV
for traj_id in unique_trajectories:
    acc_traj_data = acc_data[acc_data['trajectory_id'] == traj_id]
    hv_traj_data = hv_data[hv_data['trajectory_id'] == traj_id]

    time_acc = acc_traj_data['Time']
    total_j_acc = acc_traj_data['Total_J']
    
    if not hv_traj_data.empty:
        # Interpolate HV data to match ACC time points
        total_j_hv = np.interp(time_acc, hv_traj_data['Time'], hv_traj_data['Total_J'])
    else:
        print(f"Warning: No HV data available for trajectory {traj_id}. Skipping.")
        continue

    # Apply exponential smoothing
    moving_avg_acc = total_j_acc.ewm(span=window_size, adjust=False).mean()
    moving_avg_hv = pd.Series(total_j_hv).ewm(span=window_size, adjust=False).mean()

    # Calculate residuals
    residuals_acc = total_j_acc - moving_avg_acc
    residuals_hv = total_j_hv - moving_avg_hv

    # Compute PSD for ACC and HV residuals
    f_acc, Pxx_den_acc = welch(residuals_acc.dropna(), fs=sample_rate, nperseg=256)
    f_hv, Pxx_den_hv = welch(residuals_hv.dropna(), fs=sample_rate, nperseg=256)
    
    # Plot PSD for ACC and HV residuals
    plt.semilogy(f_acc, Pxx_den_acc, color='blue', alpha=0.6, linewidth=1, label='AV PSD' if traj_id == unique_trajectories[0] else "")
    plt.semilogy(f_hv, Pxx_den_hv, color='orange', alpha=0.6, linewidth=1, linestyle='--', label='HV PSD' if traj_id == unique_trajectories[0] else "")

# Add plot details
plt.xlabel("Frequency (Hz)", fontsize=14)
plt.ylabel("Power Spectral Density", fontsize=14)
plt.title("PSD of Residuals for All Trajectories (AV vs HV)", fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.show()