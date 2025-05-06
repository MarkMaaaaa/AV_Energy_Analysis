# **AV Energy Analysis**  
This repository contains the dataset and code for our study on automated vehicle (AV) energy consumption modeling. The AV-Micro model, specifically designed to improve energy consumption prediction for AVs, is implemented here alongside traditional VT-Micro and ARRB models.

## **Repository Contents**  
- **`Calibrate_code/`** – Contains scripts for calibrating the AV-Micro model and comparing it with traditional VT-Micro and ARRB models.  
- **`Obd_data/`** – Includes the collected On-Board Diagnostics (OBD) data from automated and human-driven vehicles, used for model development and validation.  
- **`README.md`** – This file, providing an overview of the repository.  

## **How to Use**  
1. Clone the repository:  
   ```
   git clone https://github.com/MarkMaaaaa/AV_Energy_Analysis.git
   cd AV_Energy_Analysis
   ```
2. Install dependencies (Python 3.8+ recommended):  
   ```
   pip install -r requirements.txt
   ```
3. Run the model calibration:  
   ```
   python Calibrate_code/run_calibration.py
   ```
4. View results and analyze energy consumption patterns using the data in `Obd_data/`.  
Here is a **simple introduction** for the dataset based on your directory and sample data:

---

## **Dataset Introduction**

This dataset contains time-series energy consumption data collected from Connected and Automated Vehicles (CAVs) and Human-driven Vehicles (HVs) under different roadway scenarios during Phase 1 of the VECTOR project. Each scenario represents a distinct road environment, including synthetic test roads and the real-world Peachtree corridor. The filenames follow a structured naming convention, where each file corresponds to a vehicle type (e.g., ACC: Adaptive Cruise Control CAV; HV: Human-driven Vehicle) and a specific test run number.

Each `.csv` file includes the following variables recorded at high temporal resolution:

* `Time`: Timestamp of the data point
* `Battery_SOC`: Battery State of Charge
* `Speed`: Vehicle speed (m/s)
* `MassAirFlow`: Air intake rate
* `Battery_J`: Cumulative battery energy used (Joules)
* `Engine_J`: Cumulative engine energy used (Joules)
* `Total_J`: Total energy consumed (Joules)

These data support baseline analysis of energy consumption patterns caused by onboard sensors (e.g., LiDAR), computational loads, and basic vehicle operation, providing a benchmark for future cooperative control evaluations.

---

Let me know if you want this formatted for a README or data dictionary as well.

## **Citing This Work**  
If you use this dataset or model, please cite our paper:  
[**An Advanced Microscopic Energy Consumption Model for Automated Vehicle:Development, Calibration, Verification**]  
Authors: [Ke Ma, Zhaohui Liang, Hang Zhou, Xiaopeng Li]  
https://arxiv.org/abs/2408.11797  

For any questions or contributions, feel free to open an issue or submit a pull request.  
