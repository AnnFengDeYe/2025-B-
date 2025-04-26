import pandas as pd
import numpy as np
import os
import glob
from datetime import timedelta

# --- 配置 ---
input_directory = '.'  # 当前目录
output_directory = '../ProcessedData/ProcessedTrainData'
file_pattern = 'P[0-9][0-9][0-9].csv'
# -------------

def calculate_jerk(filepath, output_dir):
    """
    读取CSV文件，计算Jerk特征，并将第一行的Jerk值设为0，然后保存到新文件。

    Args:
        filepath (str): 输入CSV文件的路径。
        output_dir (str): 保存处理后文件的目录。
    """
    try:
        print(f"正在处理文件: {filepath} ...")
        # 读取CSV文件
        df = pd.read_csv(filepath)

        # 检查必要的列是否存在
        required_columns = ['time', 'x', 'y', 'z']
        if not all(col in df.columns for col in required_columns):
            print(f"  警告: 文件 {filepath} 缺少必要的列 ({', '.join(required_columns)})。跳过此文件。")
            return

        # 处理空文件或只有一行的文件
        if df.shape[0] < 2:
            # 对于只有一行或空文件，如果需要输出文件，可以在这里创建含Jerk列（值为0或NaN）的文件
            # 但按当前逻辑，无法计算差值，所以跳过是合理的。
            # 如果需要输出，需要决定如何处理这种情况。
            print(f"  信息: 文件 {filepath} 的行数少于2行，无法计算差值。将直接复制（如果需要）或跳过。")
            # 如果需要复制并添加0值的Jerk列：
            # if df.shape[0] == 1:
            #     df['Jerk_x'] = 0.0
            #     df['Jerk_y'] = 0.0
            #     df['Jerk_z'] = 0.0
            #     filename = os.path.basename(filepath)
            #     output_filepath = os.path.join(output_dir, filename)
            #     df.to_csv(output_filepath, index=False, float_format='%.8f')
            #     print(f"  文件只有一行，已添加Jerk=0并保存到: {output_filepath}")
            # else: # 文件为空
            #     print(f"  文件为空，跳过。")
            # 目前选择跳过少于2行的文件
            return


        # --- 时间处理 ---
        df['timedelta'] = pd.to_timedelta(df['time'], errors='coerce')
        if df['timedelta'].isnull().any():
             print(f"  警告: 文件 {filepath} 包含无效的时间格式。将尝试继续处理，但结果可能不准确。")
             # df = df.dropna(subset=['timedelta']) # 如果想删除包含无效时间的行
             # if df.shape[0] < 2:
             #    print(f"  警告: 删除无效时间后，文件 {filepath} 行数少于2行。跳过。")
             #    return

        time_diffs = df['timedelta'].diff()
        rollovers = (time_diffs < pd.Timedelta(0))
        day_offset = rollovers.cumsum()
        df['absolute_timedelta'] = df['timedelta'] + pd.to_timedelta(day_offset, unit='D')
        delta_t = df['absolute_timedelta'].diff().dt.total_seconds()

        # --- 加速度差值计算 ---
        delta_x = df['x'].diff()
        delta_y = df['y'].diff()
        delta_z = df['z'].diff()

        # --- Jerk 计算 ---
        # 计算 Jerk 值，此时第一行会是 NaN
        df['Jerk_x'] = delta_x / delta_t
        df['Jerk_y'] = delta_y / delta_t
        df['Jerk_z'] = delta_z / delta_t

        # --- 修改：将第一行的 Jerk 值设置为 0 ---
        if not df.empty: # 确保 DataFrame 不是空的
             # 使用 .loc 通过行索引标签（通常是 0）和列名列表来设置值
             df.loc[df.index[0], ['Jerk_x', 'Jerk_y', 'Jerk_z']] = 0.0
        # ------------------------------------------

        # 清理临时列
        df = df.drop(columns=['timedelta', 'absolute_timedelta'])

        # --- 保存结果 ---
        filename = os.path.basename(filepath)
        output_filepath = os.path.join(output_dir, filename)
        df.to_csv(output_filepath, index=False, float_format='%.8f')
        print(f"  成功处理并保存到: {output_filepath}")

    except pd.errors.EmptyDataError:
        print(f"  警告: 文件 {filepath} 是空的。跳过此文件。")
    except Exception as e:
        print(f"  错误: 处理文件 {filepath} 时发生错误: {e}")

# --- 主程序 ---
if __name__ == "__main__":
    # 创建输出目录 (如果不存在)
    os.makedirs(output_directory, exist_ok=True)
    print(f"输出目录 '{output_directory}' 已准备就绪。")

    # 构建完整的文件搜索模式
    search_pattern = os.path.join(input_directory, file_pattern)
    print(f"正在搜索文件，模式: {search_pattern}")

    # 查找所有匹配的文件
    csv_files = glob.glob(search_pattern)

    if not csv_files:
        print("错误：在当前目录下未找到匹配 'P[0-9][0-9][0-9].csv' 格式的文件。请检查文件名和脚本位置。")
    else:
        print(f"找到 {len(csv_files)} 个文件进行处理。")
        # 遍历并处理每个文件
        for file in sorted(csv_files): # 按文件名排序处理
            calculate_jerk(file, output_directory)

    print("\n所有文件处理完成。")