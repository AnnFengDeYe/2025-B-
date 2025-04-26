import pandas as pd
import numpy as np
import os
import re
from datetime import timedelta
import glob # 引入glob模块来查找文件

def preprocess_sensor_data(input_csv_path, output_dir):
    """
    根据指定规则预处理单个传感器数据CSV文件。
    修改：
    1. MET为空的行将被丢弃。
    2. 去除连续的 x, y, z 完全重复的行（保留首行）。
    3. 添加 magnitude 列 (sqrt(x^2+y^2+z^2) - 1)。
    4. 时间列在输出时只保留 HH:MM:SS.ffffff 部分。

    Args:
        input_csv_path (str): 输入CSV文件的路径。
        output_dir (str): 保存处理后CSV文件的目录。

    Returns:
        bool: 如果处理成功则返回 True，否则返回 False。
    """
    # --- 1. 设置: 定义路径 ---
    base_filename = os.path.basename(input_csv_path)
    output_csv_path = os.path.join(output_dir, base_filename)

    print(f"--- 开始预处理文件: {input_csv_path} ---")

    # --- 2. 读取CSV数据 ---
    try:
        df = pd.read_csv(input_csv_path, dtype={'time': str})
        print(f"成功从 {input_csv_path} 读取 {len(df)} 行数据")
    except FileNotFoundError:
        print(f"错误：输入文件未找到于 {input_csv_path}")
        return False
    except Exception as e:
        print(f"读取CSV文件 {input_csv_path} 时出错: {e}")
        return False

    # --- 3. 预处理 'time' 列 (Keep as datetime for processing) ---
    print("正在处理 'time' 列 (使用完整日期时间进行计算)...")
    # Convert to datetime objects for calculations
    df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')

    # Handle potential NaT values from conversion errors before sorting
    initial_nat_count = df['time'].isnull().sum()
    if initial_nat_count > 0:
        print(f"警告：转换时发现 {initial_nat_count} 个无效的时间格式，将尝试填充或丢弃。")
        # Option 1: Drop rows with NaT right away if they can't be handled
        # df.dropna(subset=['time'], inplace=True)
        # Option 2: Try to fill later (as the original code does)

    # Sort by the full datetime
    df = df.sort_values(by='time').reset_index(drop=True)

    # --- Attempt to fill NaT gaps based on 10ms interval ---
    expected_time = df['time'].shift(1) + timedelta(milliseconds=10)
    # Mask for original NaT values *after* sorting
    original_nat_mask = df['time'].isna()
    # Try filling only the original NaT positions
    df.loc[original_nat_mask, 'time'] = expected_time[original_nat_mask]

    # --- Check for 10ms intervals and remaining NaNs ---
    time_diff = df['time'].diff()
    expected_diff = timedelta(milliseconds=10)
    # Rows to keep: the first row OR rows with the expected 10ms difference
    rows_to_keep_mask = (df.index == 0) | (time_diff == expected_diff)
    # Crucially, also ensure the time is not NaT after the potential filling attempt
    rows_to_keep_mask = rows_to_keep_mask & df['time'].notna()

    original_row_count_before_time = len(df)
    df = df[rows_to_keep_mask].reset_index(drop=True)
    rows_dropped_time = original_row_count_before_time - len(df)
    print(f"因时间间隔不一致、格式错误或无法填充的间隙，丢弃了 {rows_dropped_time} 行数据。")

    if df.empty:
        print("错误：时间处理后没有剩余有效数据。跳过此文件。")
        return False

    # --- 4. 预处理 'x', 'y', 'z' 列 ---
    print("正在处理 'x', 'y', 'z' 列...")
    for col in ['x', 'y', 'z']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    nan_counts_xyz = df[['x', 'y', 'z']].isnull().sum()
    print(f"填充前 'x' 列的空值数量: {nan_counts_xyz['x']}")
    print(f"填充前 'y' 列的空值数量: {nan_counts_xyz['y']}")
    print(f"填充前 'z' 列的空值数量: {nan_counts_xyz['z']}")

    df[['x', 'y', 'z']] = df[['x', 'y', 'z']].interpolate(method='linear', limit_direction='both')

    if df[['x', 'y', 'z']].isnull().any().any():
        print("警告：插值后 x, y, 或 z 列仍存在 NaN。这些行可能无法计算幅值或影响去重。正在丢弃这些行。")
        df.dropna(subset=['x', 'y', 'z'], inplace=True) # Drop rows if interpolation failed

    if df.empty:
        print("错误：插值或丢弃NaN后没有剩余有效数据。跳过此文件。")
        return False

    # --- 5. 添加 'magnitude' 列 ---
    print("正在计算并添加 'magnitude' 列...")
    # Ensure calculation happens *after* NaNs in x, y, z are handled
    df['magnitude'] = np.sqrt(df['x']**2 + df['y']**2 + df['z']**2)
    mag_nan_count = df['magnitude'].isnull().sum()
    if mag_nan_count > 0:
       # This should ideally not happen if NaNs in x,y,z were dropped
       print(f"警告：有 {mag_nan_count} 行未能成功计算 'magnitude'。检查 x,y,z 处理。")


    # --- 6. 去除连续重复的 x, y, z 行 ---
    print("正在去除连续重复的 'x', 'y', 'z' 行...")
    # Ensure comparison happens on cleaned numeric data
    is_duplicate_xyz = (df['x'] == df['x'].shift()) & \
                       (df['y'] == df['y'].shift()) & \
                       (df['z'] == df['z'].shift())

    rows_before_dedup = len(df)
    df = df[~is_duplicate_xyz].reset_index(drop=True)
    rows_after_dedup = len(df)
    rows_dropped_dedup = rows_before_dedup - rows_after_dedup
    print(f"因 x, y, z 值与上一行连续相同，去除了 {rows_dropped_dedup} 行重复数据。")

    # --- 7. 预处理 'annotation' 列 (提取并丢弃空值行) ---
    print("正在处理 'annotation' 列...")
    if df.empty:
        print("错误：去重后没有剩余有效数据。跳过此文件。")
        return False

    # Ensure 'annotation' column exists before processing
    if 'annotation' not in df.columns:
        print("错误：CSV 文件中缺少 'annotation' 列。跳过此文件。")
        return False

    annotation_series = df['annotation'].astype(str)
    # Use regex to extract MET value
    met_values = annotation_series.str.extract(r'MET\s+(\d+\.?\d*)', expand=False)
    # Convert extracted values to numeric, forcing errors to NaN
    df['annotation'] = pd.to_numeric(met_values, errors='coerce')

    nan_met_count = df['annotation'].isnull().sum()
    print(f"将因'annotation'列为空或无法提取MET值而丢弃的行数: {nan_met_count}")

    original_row_count_before_met_drop = len(df)
    # Drop rows where 'annotation' is NaN (either originally empty or failed extraction/conversion)
    df.dropna(subset=['annotation'], inplace=True)
    rows_dropped_met = original_row_count_before_met_drop - len(df)

    # Recalculate nan_met_count based on the state *before* dropping for accurate comparison
    # This comparison might be less critical now, focus on the outcome.
    print(f"已丢弃 {rows_dropped_met} 行 'annotation' (MET) 为空或无效的数据。")


    if df.empty:
        print("错误：丢弃空/无效MET值后没有剩余有效数据。跳过此文件。")
        return False

    # --- 8. 格式化 'time' 列并保存 ---
    print("最终格式化 'time' 列为 HH:MM:SS.ffffff ...")
    # **** Modification: Format the 'time' column to string H:M:S.f before saving ****
    try:
        df['time'] = df['time'].dt.strftime('%H:%M:%S.%f')
    except AttributeError:
        print("错误：无法格式化'time'列。可能它不是预期的datetime类型。")
        return False # Or handle appropriately

    # --- Reorder columns if needed ---
    cols = df.columns.tolist()
    # Example desired order: time, x, y, z, magnitude, annotation
    desired_order = ['time', 'x', 'y', 'z', 'magnitude', 'annotation']
    # Filter to keep only existing columns in the desired order
    final_cols = [col for col in desired_order if col in df.columns]
    # Add any other columns not in the desired list (though unlikely with this script)
    final_cols.extend([col for col in df.columns if col not in final_cols])
    df = df[final_cols] # Apply the final column order


    print(f"当前剩余有效数据行数: {len(df)}")
    print(f"最终数据列: {df.columns.tolist()}")
    print(f"正在将处理后的数据保存到: {output_csv_path}")
    try:
        # Save without date_format since 'time' is now a string
        df.to_csv(output_csv_path, index=False)
        print(f"--- 文件 {base_filename} 预处理完成。 ---")
        return True
    except Exception as e:
        print(f"保存处理后的文件到 {output_csv_path} 时出错: {e}")
        return False

# --- 主执行逻辑 ---
if __name__ == "__main__":
    # 定义输入目录（当前目录）和输出目录
    input_directory = './OriginalData' # 原始数据存放的目录
    # Make sure the output directory is correct relative to where the script is RUN
    script_dir = os.path.dirname(__file__) if '__file__' in locals() else '.'
    output_directory = os.path.join(script_dir, '.') # 输出目录

    # --- 确保输出目录存在 ---
    try:
        os.makedirs(output_directory, exist_ok=True)
        print(f"确保输出目录 '{os.path.abspath(output_directory)}' 存在或已创建。")
    except OSError as e:
        print(f"创建输出目录 '{output_directory}' 时出错: {e}。脚本将退出。")
        exit() # 如果无法创建输出目录，则退出

    # --- 查找所有符合模式 'P###.csv' 的文件 ---
    # Use input_directory which is '.' unless changed
    file_pattern = os.path.join(input_directory, 'P[0-9][0-9][0-9].csv')
    # Use absolute path for glob if input_directory is relative for clarity
    abs_input_dir = os.path.abspath(input_directory)
    csv_files_to_process = glob.glob(os.path.join(abs_input_dir, 'P[0-9][0-9][0-9].csv'))


    # --- 对文件列表进行排序 ---
    # Sort based on the filename found by glob
    csv_files_to_process.sort()

    if not csv_files_to_process:
        print(f"在目录 '{abs_input_dir}' 中未找到符合 'P###.csv' 模式的文件。")
    else:
        print(f"在目录 '{abs_input_dir}' 中找到 {len(csv_files_to_process)} 个待处理的文件 (已排序):")
        # Print just the base filename for brevity
        for f_path in csv_files_to_process:
            print(f"  - {os.path.basename(f_path)}")
        print("-" * 40) # 分隔符

        success_count = 0
        failure_count = 0

        # --- 循环处理每个找到的文件 (现在是按顺序) ---
        for input_file_path in csv_files_to_process:
            print(f"\n>>> 开始处理文件: {os.path.basename(input_file_path)} <<<\n")
            try:
                # Pass the absolute path of the input file
                success = preprocess_sensor_data(input_file_path, output_directory)
                if success:
                    success_count += 1
                else:
                    failure_count += 1
            except Exception as e:
                print(f"!!! 处理文件 {os.path.basename(input_file_path)} 时发生严重意外错误: {e} !!!")
                # Optionally print traceback for debugging:
                # import traceback
                # traceback.print_exc()
                failure_count += 1
            print("-" * 40) # 每个文件处理后的分隔符

        # --- 打印总结信息 ---
        print("\n==================== 处理总结 ====================")
        print(f"总共尝试处理文件数量: {len(csv_files_to_process)}")
        print(f"成功处理文件数量: {success_count}")
        print(f"处理失败或跳过文件数量: {failure_count}")
        print(f"处理后的文件已保存至目录: {os.path.abspath(output_directory)}")
        print("==================================================")