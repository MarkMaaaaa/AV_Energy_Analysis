import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.ticker as ticker
from scipy.stats import f
import warnings
import joblib
pd.options.mode.chained_assignment = None  # default='warn'
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

def prepare_data_ARRB(df):
    # 生成新的列
    df['speed_2'] = df['Speed'] ** 2  # Speed_smoothed
    df['speed_3'] = df['Speed'] ** 3
    df['speed_acceleration'] = df['Speed'] * df['Acceleration']
    df['speed_acceleration_squared'] = df['Speed'] * (df['positive_acc'] ** 2)

    df['p_speed_acceleration'] = df['Speed'] * df['positive_acc']
    df['p_speed_acceleration_squared'] = df['Speed'] * (df['positive_acc'] ** 2)
    df['n_speed_acceleration'] = df['Speed'] * df['negative_acc']
    df['n_speed_acceleration_squared'] = df['Speed'] * (df['negative_acc'] ** 2)

    df['last_speed_2'] = df['last_Speed'] ** 2
    df['last_speed_3'] = df['last_Speed'] ** 3
    df['last_speed_acceleration'] = df['last_Speed'] * df['last_Acceleration']
    df['last_speed_acceleration_squared'] = df['last_Speed'] * (df['last_positive_acc'] ** 2)

    # df['e_speed'] = np.exp(df['Speed'] * 1e-1)
    # df['e_speed_2'] = np.exp(df['Speed'] ** 2 * 1e-2)
    # df['e_speed_3'] = np.exp(df['Speed'] ** 3 * 1e-3)
    # df['e_speed_acceleration'] = np.exp(df['Speed'] * df['Acceleration'] * 1e-1)
    # df['e_speed_acceleration_squared'] = np.exp(df['Speed'] * (df['positive_acc'] ** 2) * 1e-1)

    # df['e_last_speed'] = np.exp(df['last_Speed'] * 1e-1)
    # df['e_last_speed_2'] = np.exp(df['last_Speed'] ** 2 * 1e-2)
    # df['e_last_speed_3'] = np.exp(df['last_Speed'] ** 3 * 1e-3)
    # df['e_last_speed_acceleration'] = np.exp(df['last_Speed'] * df['last_Acceleration'] * 1e-1)
    # df['e_last_speed_acceleration_squared'] = np.exp(df['last_Speed'] * (df['last_positive_acc'] ** 2) * 1e-1)

    df.to_csv('calibration_ARRB.csv', index=False)

    # 准备特征和目标变量
    X = df[
        ['Speed', 'speed_2', 'speed_3', 'speed_acceleration', 'speed_acceleration_squared',
         # 'last_Speed', 'last_speed_2', 'last_speed_3', 'last_speed_acceleration', 'last_speed_acceleration_squared',
         # 'e_speed', 'e_speed_2', 'e_speed_3', 'e_speed_acceleration', 'e_speed_acceleration_squared',
         # 'e_last_speed', 'e_last_speed_2', 'e_last_speed_3', 'e_last_speed_acceleration', 'e_last_speed_acceleration_squared',
         ]]
    y = df['Total_J']

    return X, y


def prepare_data_VT_micro(df):
    # 生成Total_J的自然对数
    # df['ln_Total_J'] = np.log(df['Total_J'] / 34200000)

    # 生成速度和加速度的0到3次幂的所有组合项
    i_num = 4
    j_num = 4
    for i in range(i_num):
        for j in range(j_num):
            df[f'speed_{i}_acc_{j}'] = df['Speed'] ** i * (df['Acceleration'] ** j)
            df[f'e_speed_{i}_acc_{j}'] = np.clip(np.exp(df['Speed'] ** i * (df['Acceleration'] ** j)), None, np.exp(5))
            
            df[f'speed_{i}_pacc_{j}'] = df['Speed'] ** i * (df['positive_acc'] ** j)
            df[f'speed_{i}_nacc_{j}'] = df['Speed'] ** i * (df['negative_acc'] ** j)
            
            df[f'last_speed_{i}_acc_{j}'] = df['last_Speed'] ** i * df['last_Acceleration'] ** j
            df[f'last_speed_{i}_pacc_{j}'] = df['last_Speed'] ** i * df['last_positive_acc'] ** j
            df[f'last_speed_{i}_nacc_{j}'] = df['last_Speed'] ** i * df['last_negative_acc'] ** j
            
   
    # 准备特征和目标变量
    X = df[
        # [f'speed_{i}_acc_{j}' for i in range(i_num) for j in range(j_num) if not (i == 0 and j == 0)]
         # [f'last_speed_{i}_acc_{j}' for i in range(i_num) for j in range(j_num) if not (i == 0 and j == 0)]
        # + [f'speed_{i}_pacc_{j}' for i in range(i_num) for j in range(j_num) if not (i == 0 and j == 0)]
        # + [f'speed_{i}_nacc_{j}' for i in range(i_num) for j in range(i_num) if not (i == 0 and j == 0)]
        # + [f'last_speed_{i}_pacc_{j}' for i in range(i_num) for j in range(i_num) if not (i == 0 and j == 0)]
        # + [f'last_speed_{i}_nacc_{j}' for i in range(4) for j in range(4) if not (i == 0 and j == 0)]
        [f'e_speed_{i}_acc_{j}' for i in range(i_num) for j in range(j_num) if not (i == 0 and j == 0)]
        ]
    # y = df['ln_Total_J']
    y = df['Total_J']

    return X, y

def prepare_data_AA_micro(df):
    i_num = 3
    j_num = 3
    for i in range(i_num):
        for j in range(j_num):
            df[f'speed_{i}_acc_{j}'] = df['Speed'] ** i * (df['Acceleration'] ** j)
            
            # df[f'e_speed_{i}_acc_{j}'] = np.clip(np.exp(df['Speed'] ** i * (df['Acceleration'] ** j)), None, np.exp(5))
            # df[f'e_speed_{i}_pacc_{j}'] = np.clip(np.exp(df['Speed'] ** i * (df['positive_acc'] ** j)), None, np.exp(5))
            
            df[f'e_speed_{i}_acc_{j}'] = np.exp(df['Speed'] ** i * (df['Acceleration'] ** j)* 1e-5)
            df[f'e_speed_{i}_pacc_{j}'] = np.exp(df['Speed'] ** i * (df['positive_acc'] ** j)* 1e-5)

            df[f'speed_{i}_pacc_{j}'] = df['Speed'] ** i * (df['positive_acc'] ** j)
            df[f'speed_{i}_nacc_{j}'] = df['Speed'] ** i * (df['negative_acc'] ** j)
            
            df[f'last_speed_{i}_acc_{j}'] = df['last_Speed'] ** i * df['last_Acceleration'] ** j
            df[f'last_speed_{i}_pacc_{j}'] = df['last_Speed'] ** i * df['last_positive_acc'] ** j
            df[f'last_speed_{i}_nacc_{j}'] = df['last_Speed'] ** i * df['last_negative_acc'] ** j
            
   
    # 准备特征和目标变量
    X = df[
        [f'speed_{i}_acc_{j}' for i in range(i_num) for j in range(j_num) if not (i == 0 and j == 0)]
        # + [f'last_speed_{i}_acc_{j}' for i in range(i_num) for j in range(j_num) if not (i == 0 and j == 0)]
        + [f'speed_{i}_pacc_{j}' for i in range(i_num) for j in range(j_num) if not (i == 0 and j == 0)]
        # + [f'speed_{i}_nacc_{j}' for i in range(i_num) for j in range(i_num) if not (i == 0 and j == 0)]
        # + [f'last_speed_{i}_pacc_{j}' for i in range(i_num) for j in range(i_num) if not (i == 0 and j == 0)]
        # + [f'last_speed_{i}_nacc_{j}' for i in range(4) for j in range(4) if not (i == 0 and j == 0)]
        + [f'e_speed_{i}_acc_{j}' for i in range(i_num) for j in range(j_num) if not (i == 0 and j == 0)]
        + [f'e_speed_{i}_pacc_{j}' for i in range(i_num) for j in range(j_num) if not (i == 0 and j == 0)]

        ]
    # y = df['ln_Total_J']
    y = df['Total_J']

    return X, y
    # 根据trajectory_id划分训练集和测试集
    unique_ids = df['trajectory_id'].unique()
    train_ids, test_ids = train_test_split(unique_ids, test_size=0.2, random_state=43)

    train_df = df[df['trajectory_id'].isin(train_ids)]
    test_df = df[df['trajectory_id'].isin(test_ids)]

    X_train = X.loc[train_df.index]
    y_train = y.loc[train_df.index]
    X_test = X.loc[test_df.index]
    y_test = y.loc[test_df.index]

    # 进行线性回归
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 预测
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # 计算R²
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)

    print("训练集R²:", r2_train)
    print("测试集R²:", r2_test)

    return model

def calibrate_model(X, y):
    # 进行线性回归
    model = LinearRegression()
    model.fit(X, y)

    # 预测
    y_pred = model.predict(X)

    # 计算R²
    r2 = r2_score(y, y_pred)

    # 计算调整后的 R²
    n = X.shape[0]  # 样本数量
    p = X.shape[1]  # 特征数量
    adjusted_r2 = 1 - ((1 - r2) * (n - 1)) / (n - p - 1)

    # print("数据集R²:", r2)
    print("数据集调整后的R²:", adjusted_r2)

    return model


def calibrate_ARRB_model(df):
    X, y = prepare_data_ARRB(df)
    model = calibrate_model(X, y)

    # 输出回归结果
    coefficients = model.coef_
    intercept = model.intercept_
    return model

    # print("ARRB模型回归系数:", coefficients)
    # print("ARRB模型截距:", intercept)


def calibrate_VT_micro_model(df):
    X, y = prepare_data_VT_micro(df)
    model = calibrate_model(X, y)

    # 输出回归结果
    coefficients = model.coef_
    intercept = model.intercept_
    return model

    # print("VT-micro模型回归系数:", coefficients)
    # print("VT-micro模型截距:", intercept)
    
def calibrate_AA_micro_model(df):
    X, y = prepare_data_AA_micro(df)
    model = calibrate_model(X, y)

    # 输出回归结果
    coefficients = model.coef_
    intercept = model.intercept_
    return model

    # print("VT-micro模型回归系数:", coefficients)
    # print("VT-micro模型截距:", intercept)

def predict_Total_J_ACC(model_AA_ACC, df_ACC):
    # Prepare data for prediction
    X, y = prepare_data_AA_micro(df_ACC)
    # Predict Total_J
    predictions = model_AA_ACC.predict(X)
    
    # Add predictions to the dataframe
    df_ACC['Predicted_Total_J'] = predictions
    
    return df_ACC

def predict_Total_J_HV(model_AA_HV, df_HV):
    # Prepare data for prediction
    X, y = prepare_data_AA_micro(df_HV)
    # Predict Total_J
    predictions = model_AA_HV.predict(X)
    
    # Add predictions to the dataframe
    df_HV['Predicted_Total_J'] = predictions
    
    return df_HV

def plot_real_vs_predicted_3(df, save_path=None):
    plt.figure(figsize=(14, 8))

    # Plot real Total_J
    plt.plot(df['Total_J'].values, label='Actual value', color='#1f77b4', linestyle='-', linewidth=6)

    # Plot predicted Total_J for AA model
    plt.plot(df['Predicted_AA_Total_J'].values, label='AA-Micro model prediction', color='#ff7f0e', linestyle='--', linewidth=6)

    # Plot predicted Total_J for VT model
    plt.plot(df['Predicted_VT_Total_J'].values, label='VT-Micro model prediction', color='#2ca02c', linestyle=':', linewidth=6)

    # Plot predicted Total_J for ARRB model
    plt.plot(df['Predicted_ARRB_Total_J'].values, label='ARRB model prediction', color='#d62728', linestyle='-.', linewidth=6)

    # plt.title('Total_J (J)', fontsize=20)
    plt.xlabel('Time (s)', fontsize=32)
    plt.ylabel('Energy Consumption (J)', fontsize=32)
    # plt.legend(fontsize=28)
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    plt.grid(True)
    if save_path:
        print(f"Saving plot to {save_path}")
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()


def plot_prediction_error_distribution(residuals_dict, model_group):
    plt.figure(figsize=(14, 8))
    line_styles = ['-', '--', '-.', ':']
    
    for idx, (group_name, residuals) in enumerate(residuals_dict.items()):
        sns.kdeplot(residuals, label=f'{group_name} Mean = {residuals.mean():.1f}, Std = {residuals.std():.1f}', linestyle=line_styles[idx % len(line_styles)], linewidth=8)

    # plt.title(f'Residuals Distribution for Model {model_group}', fontsize=20)
    plt.xlabel('Residuals (J)', fontsize=34)
    plt.ylabel('Probability Density', fontsize=34)
    plt.legend(fontsize=26, loc='center')
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:.0e}'.format(x)))  # 设置纵坐标为科学计数法
    plt.grid(True)
    plt.show()

def calculate_f_p_value(df, model, groups):
    rss_total = 0
    for group_name, group_data in groups.items():
        if group_data.shape[0] == 0:
            continue
        df_with_predictions = predict_Total_J_ACC(model, group_data.copy())
        residuals = df_with_predictions['Total_J'] - df_with_predictions['Predicted_Total_J']
        rss_total += np.sum(residuals ** 2)
    
    rss_model = np.sum((df['Total_J'] - df['Total_J'].mean()) ** 2) - rss_total
    df_between = len(groups) - 1
    df_within = len(df) - len(groups)
    
    msr = rss_model / df_between
    mse = rss_total / df_within
    f_value = msr / mse
    p_value = f.sf(f_value, df_between, df_within)
    
    return f_value, p_value

def calculate_rmse_mae(residuals):
    mse = np.mean(residuals ** 2)
    mae = np.mean(np.abs(residuals))
    rmse = np.sqrt(mse)
    return rmse, mae

def analyze_groups(df, threshold=0.06):
    # Group the data into three groups
    groups = {
        'Group 1': df[df['trajectory_id'].isin([0, 3, 6])],
        'Group 2': df[df['trajectory_id'].isin([1, 4, 7])],
        'Group 3': df[df['trajectory_id'].isin([2, 5, 8])]
    }
    models = {}

    # Calibrate models for each group
    for group_name, group_data in groups.items():
        X, y = prepare_data_AA_micro(group_data)
        model = calibrate_model(X, y)
        models[group_name] = model

    # Predict and calculate residuals for each group, then perform F-tests
    for model_group_name, model in models.items():
        print(f"\nResults for model trained on {model_group_name}:")

        residuals_dict = {}
        rmse_mae_dict = {}
        rss_dict = {}

        # Calculate the model's performance on each group
        for test_group_name, test_group_data in groups.items():
            df_with_predictions = predict_Total_J_ACC(model, test_group_data.copy())
            residuals = df_with_predictions['Total_J'] - df_with_predictions['Predicted_Total_J']
            
            # Apply the threshold adjustment correctly
            threshold_value = threshold * df_with_predictions['Total_J']
            adjusted_residuals = np.where(np.abs(residuals) <= np.abs(threshold_value), 0, residuals)
            residuals_dict[test_group_name] = adjusted_residuals

            # Calculate RSS based on adjusted residuals
            rss_group_total = np.sum(adjusted_residuals ** 2)
            rss_dict[test_group_name] = rss_group_total

            # Calculate RMSE and MAE based on adjusted residuals
            rmse, mae = calculate_rmse_mae(adjusted_residuals)
            rmse_mae_dict[test_group_name] = (rmse, mae)

        # Print RMSE and MAE for each test group
        for key, (rmse, mae) in rmse_mae_dict.items():
            print(f"{key}: RMSE = {rmse:.3f}, MAE = {mae:.3f}")

        # Perform F-tests between the different groups using the same model
        group_names = list(rss_dict.keys())
        for i in range(len(group_names)):
            for j in range(i + 1, len(group_names)):
                group1, group2 = group_names[i], group_names[j]
                rss1, rss2 = rss_dict[group1], rss_dict[group2]

                # Calculate degrees of freedom
                n1, n2 = len(groups[group1]), len(groups[group2])
                df1, df2 = n1 - 1, n2 - 1

                # Calculate F-value (comparing RSS adjusted by threshold)
                f_value =  (rss1 / df1)/(rss2 / df2)  if rss2 != 0 else np.inf
                p_value = f.sf(f_value, df1, df2)

                print(f"F-test between {group1} and {group2} for {model_group_name} model: F-value = {f_value:.5f}, p-value = {p_value:.5f}")

        # Optionally, plot the residuals distribution
        plot_prediction_error_distribution(residuals_dict, model_group_name)




def remove_outliers(df, column, n_std=3):
    mean = df[column].mean()
    std = df[column].std()
    return df[(df[column] >= mean - n_std * std) & (df[column] <= mean + n_std * std)]

def plot_error_over_time(df):
    plt.figure(figsize=(14, 8))

    # Calculate residuals for each model
    df['Residual_AA'] = df['Total_J'] - df['Predicted_AA_Total_J']
    df['Residual_VT'] = df['Total_J'] - df['Predicted_VT_Total_J']
    df['Residual_ARRB'] = df['Total_J'] - df['Predicted_ARRB_Total_J']

    # Plot the residuals over time
    sns.kdeplot(df['Residual_AA'], label='AA-Micro model Residuals', color='#ff7f0e', linestyle='--', linewidth=6)
    sns.kdeplot(df['Residual_VT'], label='VT-Micro model Residuals', color='#2ca02c', linestyle=':', linewidth=6)
    sns.kdeplot(df['Residual_ARRB'], label='ARRB model Residuals', color='#d62728', linestyle='-.', linewidth=6)

    plt.xlabel('Time (s)', fontsize=32)
    plt.ylabel('Residuals (J)', fontsize=32)
    plt.legend(fontsize=24)
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    plt.gca().xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    plt.gca().xaxis.get_major_formatter().set_scientific(True)
    plt.gca().xaxis.get_major_formatter().set_powerlimits((-1, 1))
    
    plt.gca().yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    plt.gca().yaxis.get_major_formatter().set_scientific(True)
    plt.gca().yaxis.get_major_formatter().set_powerlimits((-1, 1))
    plt.gca().xaxis.get_offset_text().set_fontsize(22)
    plt.gca().yaxis.get_offset_text().set_fontsize(22)
    
    plt.grid(True)
    plt.show()
    
def analyze_HV_groups_with_ACC_model(df_ACC, df_HV, model_ACC, threshold=0.06):
    # Group the HV data into three groups
    groups = {
        'Group 1': df_HV[df_HV['trajectory_id'].isin([6])],
        'Group 2': df_HV[df_HV['trajectory_id'].isin([1, 4, 7])],
        'Group 3': df_HV[df_HV['trajectory_id'].isin([2, 5, 8])]
    }

    residuals_dict = {}
    rss_dict = {}
    rmse_mae_dict = {}

    # Calculate residuals for ACC data using the ACC model
    df_ACC_with_predictions = predict_Total_J_ACC(model_ACC, df_ACC.copy())
    residuals_ACC = df_ACC_with_predictions['Total_J'] - df_ACC_with_predictions['Predicted_Total_J']
    adjusted_residuals_ACC = np.where(np.abs(residuals_ACC) <= threshold * df_ACC_with_predictions['Total_J'], 0, residuals_ACC)
    residuals_dict['ACC'] = adjusted_residuals_ACC

    # Calculate residuals for each HV group using the ACC model
    for group_name, group_data in groups.items():
        df_with_predictions = predict_Total_J_ACC(model_ACC, group_data.copy())
        residuals = df_with_predictions['Total_J'] - df_with_predictions['Predicted_Total_J']
        adjusted_residuals = np.where(np.abs(residuals) <= threshold * df_with_predictions['Total_J'], 0, residuals)
        residuals_dict[group_name] = adjusted_residuals

        # Calculate RSS
        rss_group_total = np.sum(adjusted_residuals ** 2)
        rss_dict[group_name] = rss_group_total

        # Calculate RMSE and MAE
        rmse = np.sqrt(np.mean(adjusted_residuals ** 2))
        mae = np.mean(np.abs(adjusted_residuals))
        rmse_mae_dict[group_name] = (rmse, mae)

    # Calculate RSS for ACC data
    rss_ACC = np.sum(adjusted_residuals_ACC ** 2)
    rss_dict['ACC'] = rss_ACC

    # Print RMSE and MAE for ACC and each HV group
    rmse_ACC = np.sqrt(np.mean(adjusted_residuals_ACC ** 2))
    mae_ACC = np.mean(np.abs(adjusted_residuals_ACC))
    rmse_mae_dict['ACC'] = (rmse_ACC, mae_ACC)

    for key, (rmse, mae) in rmse_mae_dict.items():
        print(f"{key}: RMSE = {rmse:.3f}, MAE = {mae:.3f}")

    # Perform F-tests between ACC and each HV group using the ACC model
    group_names = list(rss_dict.keys())
    for i in range(len(group_names)):
        for j in range(i + 1, len(group_names)):
            group1, group2 = group_names[i], group_names[j]
            rss1, rss2 = rss_dict[group1], rss_dict[group2]

            # Calculate degrees of freedom
            n1, n2 = len(df_ACC if group1 == 'ACC' else groups[group1]), len(df_ACC if group2 == 'ACC' else groups[group2])
            df1, df2 = n1 - 1, n2 - 1

            # Directly calculate F-value and p-value
            f_value = (rss2 / df2) / (rss1 / df1) if rss2 != 0 else np.inf
            p_value = f.sf(f_value, df1, df2)

            print(f"F-test between {group1} and {group2} for ACC model: F-value = {f_value:.5f}, p-value = {p_value:.5f}")

    # Plot the residuals for ACC and each HV group
    plot_residuals_distribution(residuals_dict)
    
def plot_residuals_distribution(residuals_dict):
    plt.figure(figsize=(14, 8))
    line_styles = ['-', '--', '-.', ':']
    
    for idx, (group_name, residuals) in enumerate(residuals_dict.items()):
        sns.kdeplot(residuals, label=f'{group_name} Mean = {residuals.mean():.1f}, Std = {residuals.std():.1f}', linestyle=line_styles[idx % len(line_styles)], linewidth=6)

    plt.xlabel('Residuals (J)', fontsize=34)
    plt.ylabel('Probability Density', fontsize=34)
    plt.legend(fontsize=24)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.grid(True)
    plt.show()
    
def calculate_energy_per_meter(df, total_energy):
    time_interval = 1  # Assuming time step of 1 second for simplicity (adjust based on your data)
    total_distance = np.sum(df['Speed'] * time_interval)
    energy_per_meter = total_energy / total_distance
    return energy_per_meter

def plot_comparison_ACC_HV_model(df, model_AA_ACC, model_AA_HV, save_path=None):
    plt.figure(figsize=(14, 8))

    # Predict using the ACC and HV models
    df_ACC_predictions = predict_Total_J_ACC(model_AA_ACC, df.copy())
    df_HV_predictions = predict_Total_J_HV(model_AA_HV, df.copy())

    # Plot real Total_J
    plt.plot(df['Total_J'].values, color='#1f77b4', linestyle='-', linewidth=6)

    # Plot predicted Total_J for ACC model
    plt.plot(df_ACC_predictions['Predicted_Total_J'].values, color='#ff7f0e', linestyle='--', linewidth=6)

    # Plot predicted Total_J for HV model
    plt.plot(df_HV_predictions['Predicted_Total_J'].values, color='#2ca02c', linestyle='-.', linewidth=6)

    plt.xlabel('Time(s)', fontsize=32)
    plt.ylabel('Energy Consumption (J)', fontsize=32)
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    plt.grid(True)
    if save_path:
        print(f"Saving plot to {save_path}")
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def print_AA_micro_coefficients(model, feature_names):
    coefficients = model.coef_
    
    # Iterate over coefficients and corresponding feature names
    for i, coef in enumerate(coefficients):
        print(f"{feature_names[i]}: {coef}")


file_number = 8
# 读取CSV文件
df_ACC = pd.read_csv('ACC_processed_output.csv')
df_HV = pd.read_csv('HV_processed_output.csv')

# df_ACC = df_ACC[(df_ACC['Speed'] > 8.941) & (df_ACC['trajectory_id'] == file_number) & ~((df_ACC['Acceleration'] == 0) & (df_ACC['Total_J'] < 25000))]
df_ACC = df_ACC[(df_ACC['Speed'] > 8.941) & (df_ACC['trajectory_id'] != -3) & ~((df_ACC['Acceleration'] == 0) & (df_ACC['Total_J'] < 25000))]
df_ACC['Acceleration'] = df_ACC['Acceleration'].rolling(window=4, min_periods=1).mean()
df_ACC['positive_acc'] = df_ACC['positive_acc'].rolling(window=4, min_periods=1).mean()
# df_ACC['Speed'] = df_ACC['Speed'].rolling(window=5, min_periods=1).mean()

# df_HV = df_HV[(df_HV['Speed'] > 8.941) & (df_HV['trajectory_id'] == file_number) & (df_HV['Acceleration'] != 0) ]
df_HV = df_HV[(df_HV['Speed'] > 8.941) & (df_HV['trajectory_id'] != -3) & (df_HV['Acceleration'] != 0) ]
df_HV['Acceleration'] = df_HV['Acceleration'].rolling(window=4, min_periods=1).mean()
df_HV['positive_acc'] = df_HV['positive_acc'].rolling(window=4, min_periods=1).mean()

print('ACC result:')
model_AA_ACC = calibrate_AA_micro_model(df_ACC)
model_VT_ACC = calibrate_VT_micro_model(df_ACC)
model_ARRB_ACC = calibrate_ARRB_model(df_ACC)

print('HV result:')
model_AA_HV = calibrate_AA_micro_model(df_HV)
model_VT_HV = calibrate_VT_micro_model(df_HV)
model_ARRB_HV = calibrate_ARRB_model(df_HV)

X_ACC_AA, y_ACC = prepare_data_AA_micro(df_ACC)
X_HV_AA, y_HV = prepare_data_AA_micro(df_HV)
# """Plot Trajectory with 3 models predictions
X_ACC_ARRB, y_ACC = prepare_data_ARRB(df_ACC)
X_ACC_VT, y_ACC = prepare_data_VT_micro(df_ACC)

X_HV_ARRB, y_HV = prepare_data_ARRB(df_HV)
X_HV_VT, y_HV = prepare_data_VT_micro(df_HV)

df_ACC_predictions_AA = model_AA_ACC.predict(X_ACC_AA)
df_ACC_predictions_VT = model_VT_ACC.predict(X_ACC_VT)
df_ACC_predictions_ARRB = model_ARRB_ACC.predict(X_ACC_ARRB)

df_ACC['Predicted_AA_Total_J'] = df_ACC_predictions_AA
df_ACC['Predicted_VT_Total_J'] = df_ACC_predictions_VT
df_ACC['Predicted_ARRB_Total_J'] = df_ACC_predictions_ARRB

df_HV_predictions_AA = model_AA_HV.predict(X_HV_AA)
df_HV_predictions_VT = model_VT_HV.predict(X_HV_VT)
df_HV_predictions_ARRB = model_ARRB_HV.predict(X_HV_ARRB)

df_HV['Predicted_AA_Total_J'] = df_HV_predictions_AA
df_HV['Predicted_VT_Total_J'] = df_HV_predictions_VT
df_HV['Predicted_ARRB_Total_J'] = df_HV_predictions_ARRB

# save_path_ACC = f"app_pre_ACC_{file_number}.png"
# plot_real_vs_predicted_3(df_ACC,save_path_ACC)
# save_path_HV = f"app_pre_HV_{file_number}.png"
# plot_real_vs_predicted_3(df_HV,save_path_HV)

# # print('coef:', model_AA_ACC.coef_)
# # print('intercept:', model_AA_ACC.intercept_)

# df_ACC = df_ACC[(df_ACC['Speed'] > 8.941) & (df_ACC['trajectory_id'] == file_number) & ~((df_ACC['Acceleration'] == 0) & (df_ACC['Total_J'] < 25000))]
# X_ACC_AA, y_ACC = prepare_data_AA_micro(df_ACC)
# df_ACC_predictions_AA_ACCdata = model_AA_ACC.predict(X_ACC_AA)
# df_HV_predictions_AA_ACCdata = model_AA_HV.predict(X_ACC_AA)
# model_ACC_predictions_AA_ACCdata = np.mean(df_ACC_predictions_AA_ACCdata)
# model_HV_predictions_AA_ACCdata = np.mean(df_HV_predictions_AA_ACCdata)
# model_real_ACCdata = np.mean(y_ACC)
# # print(f"Real ACC Data Value: {model_real_ACCdata}")
# # print(f"ACC Model HV Data Value: {model_ACC_predictions_AA_ACCdata}")
# # print(f"HV Model HV Data Value: {model_HV_predictions_AA_ACCdata}")

# real_energy_per_meter_ACC = calculate_energy_per_meter(df_ACC, model_real_ACCdata)
# acc_model_energy_per_meter_ACC = calculate_energy_per_meter(df_ACC, model_ACC_predictions_AA_ACCdata)
# hv_model_energy_per_meter_ACC = calculate_energy_per_meter(df_ACC, model_HV_predictions_AA_ACCdata)

# # Output results
# print(f"Real ACC Data Value (J/m): {real_energy_per_meter_ACC}")
# print(f"ACC Model ACC Data Value (J/m): {acc_model_energy_per_meter_ACC}")
# print(f"HV Model ACC Data Value (J/m): {hv_model_energy_per_meter_ACC}")

# save_path_ACC = f"app_com_ACC_{file_number}.png"
# plot_comparison_ACC_HV_model(df_ACC, model_AA_ACC, model_AA_HV, save_path_ACC)

# df_HV = df_HV[(df_HV['Speed'] > 8.941) & (df_HV['trajectory_id'] == file_number) & (df_HV['Acceleration'] != 0) ]
# X_HV_AA, y_HV = prepare_data_AA_micro(df_HV)
# df_ACC_predictions_AA_HVdata = model_AA_ACC.predict(X_HV_AA)
# df_HV_predictions_AA_HVdata = model_AA_HV.predict(X_HV_AA)
# model_ACC_predictions_AA_HVdata = np.mean(df_ACC_predictions_AA_HVdata)
# model_HV_predictions_AA_HVdata = np.mean(df_HV_predictions_AA_HVdata)
# model_real_HVdata = np.mean(y_HV)
# # print(f"Real HV Data Value: {model_real_HVdata}")
# # print(f"ACC Model HV Data Value: {model_ACC_predictions_AA_HVdata}")
# # print(f"HV Model HV Data Value: {model_HV_predictions_AA_HVdata}")

# real_energy_per_meter_HV = calculate_energy_per_meter(df_HV, model_real_HVdata)
# acc_model_energy_per_meter_HV = calculate_energy_per_meter(df_HV, model_ACC_predictions_AA_HVdata)
# hv_model_energy_per_meter_HV = calculate_energy_per_meter(df_HV, model_HV_predictions_AA_HVdata)

# # Output results
# print(f"Real HV Data Value (J/m): {real_energy_per_meter_HV}")
# print(f"ACC Model HV Data Value (J/m): {acc_model_energy_per_meter_HV}")
# print(f"HV Model HV Data Value (J/m): {hv_model_energy_per_meter_HV}")

# save_path_HV = f"app_com_HV_{file_number}.png"
# plot_comparison_ACC_HV_model(df_HV, model_AA_ACC, model_AA_HV, save_path_HV)




# plot_error_over_time(df_ACC)
# plot_error_over_time(df_HV)

# """
# analyze_groups(df_ACC)
# analyze_HV_groups_with_ACC_model(df_ACC, df_HV, model_AA_ACC)
# df_ACC_with_predictions = predict_Total_J_ACC(model_AA_ACC, df_ACC)
# print(df_ACC_with_predictions[['Speed', 'Acceleration', 'Total_J', 'Predicted_Total_J']].head())

# df_HV_with_predictions = predict_Total_J_HV(model_AA_HV, df_ACC)
# print(df_ACC_with_predictions[['Speed', 'Acceleration', 'Total_J', 'Predicted_Total_J']].head())


# plot_prediction_error_distribution(df_ACC)
# plot_prediction_error_distribution(df_HV)

"""Calculate stastics
sse = np.sum((df_ACC_with_predictions['Total_J'] - df_ACC_with_predictions['Predicted_Total_J']) ** 2)
mae = np.mean(np.abs(df_HV['Total_J'] - df_HV['Predicted_Total_J']))
mse = np.mean((df_HV['Total_J'] - df_HV['Predicted_Total_J']) ** 2)
rmse = np.sqrt(mse)
r2 = r2_score(df_HV['Total_J'], df_HV['Predicted_Total_J'])
print('mean_ACC_real_J:', np.mean(df_ACC['Total_J']))
# print('mean_ACC_predicted_J:', np.mean(df_ACC['df_ACC_predictions']))
print('mean_HV_real_J:', np.mean(df_HV['Total_J']))
# print('mean_HV_predicted_J:', np.mean(df_HV['df_HV_predictions']))
# print('SSE:',sse)
print('MAE:', mae)
print('MSE:', mse)
print('RMSE:', rmse)
print('R²:', r2)
"""




# average_total_j_per_trajectory_human = df_HV.groupby('trajectory_id')['Total_J'].mean().reset_index()
""" Plot acceleration==0 distribution
filtered_df_ACC = df_ACC[df_ACC['Acceleration'] == 0]
plt.figure(figsize=(10, 6))
sns.histplot(filtered_df_ACC['Total_J'], kde=True, color='blue', bins=30)
plt.title('Distribution of Total_J for Zero Acceleration')
plt.xlabel('Total_J')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
"""

# 打包保存模型和特征数据
# joblib.dump((model_AA_ACC, X_ACC_AA), 'model_and_data_AA_ACC.pkl')



