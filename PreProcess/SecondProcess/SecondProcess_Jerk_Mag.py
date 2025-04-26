import pandas as pd
import numpy as np # 需要 numpy 来进行数学计算 (sqrt)
import os
import glob
from datetime import timedelta

# --- 配置 ---
input_directory = './OriginalTestData'  # 原始测试数据集所在目录
output_directory = '../ProcessedTrainData'
file_pattern = 'P[0-9][0-9][0-9].csv'
# -------------

def calculate_jerk(filepath, output_dir):
    """
    读取CSV文件，处理时间格式，计算magnitude(如果需要)，
    计算Jerk特征，并将第一行的Jerk值设为0，然后保存到新文件。

    Args:
        filepath (str): 输入CSV文件的路径。
        output_dir (str): 保存处理后文件的目录。
    """
    try:
        print(f"正在处理文件: {filepath} ...")
        # 读取CSV文件
        # 使用 low_memory=False 减少混合类型警告的可能性
        df = pd.read_csv(filepath, low_memory=False)

        # 检查必要的列是否存在 (现在只需要 time, x, y, z)
        required_columns = ['time', 'x', 'y', 'z']
        if not all(col in df.columns for col in required_columns):
            print(f"  警告: 文件 {filepath} 缺少必要的列 ({', '.join(required_columns)})。跳过此文件。")
            return

        # 处理空文件或只有一行的文件
        if df.shape[0] < 2:
            print(f"  信息: 文件 {filepath} 的行数少于2行，无法计算差值。跳过此文件。")
            # 可以在此处添加逻辑，为单行文件添加 magnitude 和 Jerk=0 列后输出
            return

        # --- 新逻辑 2: 时间格式处理 ---
        # 尝试确定时间列的格式，只检查第一个有效条目
        first_valid_time = df['time'].dropna().iloc[0] if not df['time'].dropna().empty else None

        if first_valid_time and isinstance(first_valid_time, str) and ' ' in first_valid_time and '-' in first_valid_time:
            # 看起来像 'YYYY-MM-DD HH:MM:SS.ffffff' 格式
            print(f"  检测到长日期时间格式，正在转换为 HH:MM:SS.ffffff ...")
            try:
                # 方法1：转换为 datetime 再格式化回字符串 (更健壮)
                df['time'] = pd.to_datetime(df['time'], errors='coerce').dt.strftime('%H:%M:%S.%f')
                # strftime 可能会将 NaT (由 coerce 产生) 变成字符串 'NaT'，需要替换回 NaN
                df['time'] = df['time'].replace('NaT', np.nan)
                # 方法2：直接字符串分割 (如果格式严格一致)
                # df['time'] = df['time'].str.split(' ').str[1]
            except Exception as e:
                print(f"  警告：尝试转换时间格式时出错: {e}。将尝试按原样处理时间列。")
        else:
            print(f"  时间格式似乎是 HH:MM:SS.ffffff 或其他格式，按原样处理。")
            # 确保时间列是字符串类型，以便后续 to_timedelta 处理
            df['time'] = df['time'].astype(str)


        # --- 新逻辑 1: Magnitude 计算 ---
        # 在计算 Magnitude 前，确保 x, y, z 是数值类型，无效值转为 NaN
        df['x'] = pd.to_numeric(df['x'], errors='coerce')
        df['y'] = pd.to_numeric(df['y'], errors='coerce')
        df['z'] = pd.to_numeric(df['z'], errors='coerce')

        if 'magnitude' not in df.columns:
            print(f"  'magnitude' 列不存在，正在计算...")
            # 计算 magnitude，如果 x, y, 或 z 是 NaN，结果也是 NaN
            df['magnitude'] = np.sqrt(df['x']**2 + df['y']**2 + df['z']**2)
        else:
            print(f"  'magnitude' 列已存在，确保其为数值类型...")
            # 如果 magnitude 列已存在，同样确保它是数值类型
            df['magnitude'] = pd.to_numeric(df['magnitude'], errors='coerce')

        # 检查在转换 x, y, z 或 magnitude 时是否产生了过多 NaN (可选)
        if df[['x', 'y', 'z']].isnull().any().any():
             print(f"  警告: x, y, 或 z 列中包含非数值数据，已转换为 NaN。")
        if 'magnitude' in df.columns and df['magnitude'].isnull().any().any():
             print(f"  警告: magnitude 列中包含非数值数据或计算源数据含NaN，已转换为 NaN。")


        # --- 时间处理 (用于 Jerk 计算) ---
        # 此时 df['time'] 应该是 'HH:MM:SS.ffffff' 格式的字符串或 NaN
        df['timedelta'] = pd.to_timedelta(df['time'], errors='coerce')

        # 检查转换后的 timedelta 是否有效
        if df['timedelta'].isnull().all():
             print(f"  错误: 无法从 'time' 列解析任何有效的时间。检查原始数据格式。跳过此文件。")
             return # 如果完全没有有效时间，无法继续
        elif df['timedelta'].isnull().any():
             print(f"  警告: 'time' 列中部分数据无法解析为时间，对应行的 Jerk 将为 NaN。")

        # 处理时间回滚和计算 delta_t (同之前逻辑)
        time_diffs = df['timedelta'].diff()
        rollovers = (time_diffs < pd.Timedelta(0))
        day_offset = rollovers.cumsum()
        df['absolute_timedelta'] = df['timedelta'] + pd.to_timedelta(day_offset, unit='D')
        delta_t = df['absolute_timedelta'].diff().dt.total_seconds()


        # --- 加速度差值计算 ---
        # x, y, z 已经是数值类型 (或 NaN)
        delta_x = df['x'].diff()
        delta_y = df['y'].diff()
        delta_z = df['z'].diff()

        # --- Jerk 计算 ---
        # 计算 Jerk 值，此时第一行会是 NaN，无效 delta_t 或 delta_accel 也会产生 NaN
        df['Jerk_x'] = delta_x / delta_t
        df['Jerk_y'] = delta_y / delta_t
        df['Jerk_z'] = delta_z / delta_t

        # --- 将第一行的 Jerk 值设置为 0 ---
        if not df.empty:
             df.loc[df.index[0], ['Jerk_x', 'Jerk_y', 'Jerk_z']] = 0.0

        # --- 清理临时列 ---
        # errors='ignore' 避免在列不存在时报错 (例如，如果文件处理早期失败)
        df = df.drop(columns=['timedelta', 'absolute_timedelta'], errors='ignore')

        # --- 保存结果 ---
        filename = os.path.basename(filepath)
        output_filepath = os.path.join(output_dir, filename)
        # 控制输出精度，避免科学计数法 (如果需要)
        df.to_csv(output_filepath, index=False, float_format='%.8f')
        print(f"  成功处理并保存到: {output_filepath}")

    except pd.errors.EmptyDataError:
        print(f"  警告: 文件 {filepath} 是空的。跳过此文件。")
    except KeyError as e:
        print(f"  错误: 文件 {filepath} 缺少必需的列: {e}。跳过此文件。")
    except Exception as e:
        print(f"  错误: 处理文件 {filepath} 时发生意外错误: {e}")
        import traceback
        traceback.print_exc() # 打印详细错误堆栈

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
        print(f"错误：在目录 '{input_directory}' 下未找到匹配 '{file_pattern}' 格式的文件。请检查文件名和脚本位置。")
    else:
        print(f"找到 {len(csv_files)} 个文件进行处理。")
        # 遍历并处理每个文件
        for file in sorted(csv_files): # 按文件名排序处理
            calculate_jerk(file, output_directory)

    print("\n所有文件处理完成。")