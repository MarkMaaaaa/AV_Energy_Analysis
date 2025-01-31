import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


def prepare_data_AA_micro(df):
    i_num = 3
    j_num = 3
    for i in range(i_num):
        for j in range(j_num):
            df[f'speed_{i}_acc_{j}'] = df['Speed'] ** i * (df['Acceleration'] ** j)
            df[f'e_speed_{i}_acc_{j}'] = np.clip(np.exp(df['Speed'] ** i * (df['Acceleration'] ** j)), None, np.exp(5))
            df[f'e_speed_{i}_pacc_{j}'] = np.clip(np.exp(df['Speed'] ** i * (df['positive_acc'] ** j)), None, np.exp(5))

            df[f'speed_{i}_pacc_{j}'] = df['Speed'] ** i * (df['positive_acc'] ** j)
            df[f'speed_{i}_nacc_{j}'] = df['Speed'] ** i * (df['negative_acc'] ** j)
            
            df[f'last_speed_{i}_acc_{j}'] = df['last_Speed'] ** i * df['last_Acceleration'] ** j
            df[f'last_speed_{i}_pacc_{j}'] = df['last_Speed'] ** i * df['last_positive_acc'] ** j
            df[f'last_speed_{i}_nacc_{j}'] = df['last_Speed'] ** i * df['last_negative_acc'] ** j
            
   
    # 准备特征和目标变量
    X = df[
        [f'speed_{i}_acc_{j}' for i in range(i_num) for j in range(j_num) if not (i == 0 and j == 0)]
         # [f'last_speed_{i}_acc_{j}' for i in range(i_num) for j in range(j_num) if not (i == 0 and j == 0)]
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

def calibrate_model(X, y, df):
    # 根据trajectory_id划分训练集和测试集
    unique_ids = df['trajectory_id'].unique()
    train_ids, test_ids = train_test_split(unique_ids, test_size=0.3, random_state=103)

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
    y_pred_total = model.predict(X)
    # 计算R²
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    r2_total = r2_score(y, y_pred_total)

    print("训练集R²:", r2_train)
    print("测试集R²:", r2_test)
    print("总体数据集R²:", r2_total)

    return model


def calibrate_AAM_model(df):
    X, y = prepare_data_AA_micro(df)
    model = calibrate_model(X, y, df)

    # 输出回归结果
    coefficients = model.coef_
    intercept = model.intercept_

    print("AAM_coef:", coefficients)
    print("AAM_inter:", intercept)



# 读取CSV文件
df_ACC = pd.read_csv('HV_processed_output.csv')
df_ACC = df_ACC[(df_ACC['Speed'] > 8.941) & (df_ACC['trajectory_id'] != -1) & ~((df_ACC['Acceleration'] == 0) & (df_ACC['Total_J'] < 25000))]
df_ACC['Acceleration'] = df_ACC['Acceleration'].rolling(window=5, min_periods=1).mean()
df_ACC['positive_acc'] = df_ACC['positive_acc'].rolling(window=5, min_periods=1).mean()
# df_ACC['Speed'] = df_ACC['Speed'].rolling(window=5, min_periods=1).mean()

# df_ACC['Total_J_smoothed'] = df_ACC['Total_J'].rolling(window=5, min_periods=1).mean()
# df_HV = df_HV[(df_HV['Speed'] > 9) & (df_HV['Acceleration'] != 0) & (df_HV['trajectory_id'] != 0)]
# df_HV['Acceleration'] = df_HV['Acceleration'].rolling(window=5, min_periods=1).mean()
# df_HV['positive_acc'] = df_HV['positive_acc'].rolling(window=5, min_periods=1).mean()# df_ACC = df_ACC[(df_ACC['Speed'] > 8.9408) & (df_ACC['Acceleration'] != 0) & (df_ACC['trajectory_id'] > -1)]
calibrate_AAM_model(df_ACC)


df_HV = pd.read_csv('HV_processed_output.csv')
df_HV = df_HV[(df_HV['Speed'] > 8.941) & (df_HV['Acceleration'] != 0) & (df_HV['trajectory_id'] != 0)]
df_HV['Acceleration'] = df_HV['Acceleration'].rolling(window=5, min_periods=1).mean()
df_HV['positive_acc'] = df_HV['positive_acc'].rolling(window=5, min_periods=1).mean()
calibrate_AAM_model(df_HV)




















