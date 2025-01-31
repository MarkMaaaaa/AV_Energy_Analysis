import pandas as pd
import os

# 定义路径和参数
base_dir = 'data\\'
scenarios = ['peachtree', 'sythentic']
sub_scenarios = ['scenario1', 'scenario2', 'scenario3']
vehicle_types = ['HV'] # , 'ACC', 'HV'
file_names = ['1.trc', '2.trc']

# 初始化一个空的DataFrame来存储所有数据
all_data = pd.DataFrame()
trajectory_id = 0

# 遍历所有文件
for scenario in scenarios:
    for sub_scenario in sub_scenarios:
        for vehicle_type in vehicle_types:
            for file_name in file_names:
                joined_str = f"{scenario}-{sub_scenario}-{vehicle_type}-{file_name}"
                final_str = joined_str[:-4]  # 去掉文件名的最后四个字符（'.trc'）
                input_path = os.path.join(base_dir, final_str + '.csv')

                # 读取CSV文件
                if os.path.exists(input_path):
                    df = pd.read_csv(input_path)
                    df['trajectory_id'] = trajectory_id
                    trajectory_id += 1

                    # 合并数据
                    all_data = pd.concat([all_data, df], ignore_index=True)
                else:
                    print(f"文件未找到: {input_path}")

# 将合并后的数据保存到一个新的CSV文件
output_path = os.path.join('HV_data.csv')
all_data.to_csv(output_path, index=False)

print(f"合并数据保存到: {output_path}")
