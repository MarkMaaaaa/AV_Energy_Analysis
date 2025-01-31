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

## **Citing This Work**  
If you use this dataset or model, please cite our paper:  
[**An Advanced Microscopic Energy Consumption Model for Automated Vehicle:Development, Calibration, Verification**]  
Authors: [Ke Ma, Zhaohui Liang, Hang Zhou, Xiaopeng Li]  
https://arxiv.org/abs/2408.11797  

For any questions or contributions, feel free to open an issue or submit a pull request.  
