import pandas as pd
import numpy as np

# 读取CSV文件
df = pd.read_csv('./ACC_data.csv')

# 速度单位转换
df['Speed'] = df['Speed'] * 0.44704  # mile/hour -> meter/second

# 处理时间
df = df.drop(columns=['Time'])
df['Time'] = np.nan
for trajectory_id in df['trajectory_id'].unique():
    traj_df = df[df['trajectory_id'] == trajectory_id].copy()

    # 更改时间列，使其从0开始，每0.1递增
    time_values = np.linspace(0, (len(traj_df) - 1) * 0.1, len(traj_df))
    # time_values = np.linspace(0, (len(traj_df) - 1) * 0.1, len(traj_df)) // 2 * 2
    df.loc[df['trajectory_id'] == trajectory_id, 'Time'] = time_values
df['Time'] = df['Time'].astype(int)

# 取一秒的平均值
columns_to_average = ['Battery_SOC', 'Speed', 'MassAirFlow', 'Battery_J', 'Engine_J', 'Total_J']
df = df.groupby(['trajectory_id', 'Time'])[columns_to_average].mean().reset_index()

# 滑动平均
df['Time_diff'] = np.nan
df['Speed_diff'] = np.nan
df['Acceleration'] = np.nan

for trajectory_id in df['trajectory_id'].unique():
    traj_df = df[df['trajectory_id'] == trajectory_id].copy()

    # 更改时间列，使其从0开始，每0.1递增
    time_values = np.linspace(0, (len(traj_df) - 1) * 0.1, len(traj_df))
    df.loc[df['trajectory_id'] == trajectory_id, 'Time'] = time_values

    # 平滑速度
    window_size = 1  # 设置窗口大小，可以根据需要调整
    speed_smoothed = traj_df['Speed'].rolling(window=window_size, min_periods=1, center=True).mean()
    df.loc[df['trajectory_id'] == trajectory_id, 'Speed_smoothed'] = speed_smoothed

    # 计算速度差分和时间差分
    time_diff = np.diff(time_values, append=np.nan)
    speed_diff = np.diff(speed_smoothed, append=np.nan)

    df.loc[df['trajectory_id'] == trajectory_id, 'Time_diff'] = time_diff
    df.loc[df['trajectory_id'] == trajectory_id, 'Speed_diff'] = speed_diff

    # 计算加速度
    acceleration = speed_diff / time_diff
    df.loc[df['trajectory_id'] == trajectory_id, 'Acceleration'] = acceleration

df = df[df['Speed'] >= 0.1]

df['positive_acc'] = df['Acceleration']
df.loc[df['positive_acc'] < 0, 'positive_acc'] = 0
df['negative_acc'] = df['Acceleration']
df.loc[df['negative_acc'] > 0, 'negative_acc'] = 0

df['last_Speed'] = df['Speed'].shift(1)
df['last_Acceleration'] = df['Acceleration'].shift(1)
df['last_positive_acc'] = df['positive_acc'].shift(1)
df['last_negative_acc'] = df['negative_acc'].shift(1)

df['Total_J'] = df['Engine_J'] - df['Battery_J']

# 去掉由于差分导致没有加速度的一行
df = df.dropna()

# 删除不需要的中间列
df = df.drop(columns=['Time_diff', 'Speed_diff'])

# # Define a matrix for calculating the VT model, a vehicular dynamics model.
# K_matrix = np.array([
#     [-7.537, 0.4438, 0.1716, -0.0420],
#     [0.0973, 0.0518, 0.0029, -0.0071],
#     [-0.003, -7.42e-04, 1.09e-04, 1.16e-04],
#     [5.3e-05, 6e-06, -1e-05, -6e-06]
# ])
#
#
# # Calculate the VT model for fuel consumption and environmental impact.
# def calculate_VT_model(v, a, K):
#     sum_j1_j2 = 0
#     for j1 in range(4):
#         for j2 in range(4):
#             sum_j1_j2 += K[j1][j2] * (v ** j1) * (a ** j2)
#     F = np.exp(sum_j1_j2)
#     return F
#
#
# # Define the Vehicle Specific Power (VSP) model for vehicular dynamics.
# def calculate_VSP(v, a):
#     return v * (1.1 * a + 0.132) + 3.02 * 10 ** (-4) * v ** 3
#
#
# def calculate_VSP_model(v, a):
#     VSP = calculate_VSP(v, a)
#     if VSP < -10:
#         return 2.48e-03
#     elif -10 <= VSP < 10:
#         return 1.98e-03 * VSP ** 2 + 3.97e-02 * VSP + 2.01e-01
#     else:
#         return 7.93e-02 * VSP + 2.48e-03
#
#
# # Define the ARRB model, another vehicular dynamics model.
# def calculate_ARRB_model(v, a):
#     return (0.666 + 0.019 * v + 0.001 * v ** 2 + 0.0005 * v ** 3 + 0.122 * a + 0.793 * max(a, 0) ** 2)
#
#
# # Process each group of data by calculating TTC.
# df['VT_micro_model'] = df.apply(
#     lambda row: calculate_VT_model(row['Speed'], row['Acceleration'], K_matrix), axis=1) * 34200000
#
# df['VSP_model'] = df.apply(lambda row: calculate_VSP_model(row['Speed'], row['Acceleration']),
#                            axis=1) / 800 * 34200000
#
# df['ARRB_model'] = df.apply(lambda row: calculate_ARRB_model(row['Speed'], row['Acceleration']),
#                             axis=1) / 1000 * 34200000

# 保存修改后的数据到一个新的CSV文件
df.to_csv('ACC_processed_output.csv', index=False)
