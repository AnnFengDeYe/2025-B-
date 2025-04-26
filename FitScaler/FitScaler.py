# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import glob
import os
from tqdm import tqdm
import pickle

# --- Metadata and Config (Keep as is) ---
try:
    from metadata import PARTICIPANT_METADATA, AGE_GROUP_TO_CODE, SEX_TO_CODE
except ImportError:
    print("警告：无法导入 metadata.py。请确保该文件存在且包含必要的变量。")
    # Default values...
    PARTICIPANT_METADATA = {'P001': {'age_group': '18-24', 'sex': 'Male'}, 'P002': {'age_group': '25-34', 'sex': 'Female'}}
    AGE_GROUP_TO_CODE = {'18-24': 0, '25-34': 1, '35-49': 2, '50+': 3}
    SEX_TO_CODE = {'Male': 0, 'Female': 1}

DATA_DIR = '.' # 预处理过后等待训练的数据集目录
NUM_SENSOR_FEATURES = 7
SENSOR_FEATURES_COLS = ['x', 'y', 'z', 'magnitude', 'Jerk_x', 'Jerk_y', 'Jerk_z']

# --- map_annotation_to_class (Keep as is) ---
def map_annotation_to_class(annotation):
    if annotation < 1.0: return 0
    elif 1.0 <= annotation < 1.6: return 1
    elif 1.6 <= annotation < 3.0: return 2
    elif 3.0 <= annotation < 6.0: return 3
    else: return 4

# --- load_preprocess_add_pid (MODIFIED) ---
def load_preprocess_add_pid(file_path):
    try:
        pid = os.path.basename(file_path).split('.')[0]
        if pid not in PARTICIPANT_METADATA:
            # print(f"警告: 元数据中未找到 PID {pid} (文件 {file_path})，跳过。")
            return None

        # print(f"--- Debug: Reading {file_path} ---") # Optional debug print
        df = pd.read_csv(file_path)

        # <<< --- START MODIFICATION: Clean Column Names --- >>>
        original_columns = list(df.columns)
        # Remove leading/trailing whitespace from column names
        df.columns = df.columns.str.strip()
        cleaned_columns = list(df.columns)
        if original_columns != cleaned_columns:
             print(f"--- Info: Cleaned column names for {file_path}. Original: {original_columns}, Cleaned: {cleaned_columns} ---")
        # Optional: If you suspect case issues, you could add .str.lower()
        # df.columns = df.columns.str.strip().str.lower()
        # If you use .str.lower(), make sure required_columns and SENSOR_FEATURES_COLS are also lowercase
        # <<< --- END MODIFICATION --- >>>

        # Now check using the cleaned column names
        required_columns = ['time', 'x', 'y', 'z', 'magnitude', 'annotation', 'Jerk_x', 'Jerk_y', 'Jerk_z']
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            # Make the warning more informative
            print(f"警告: 文件 {file_path} 缺少必需列 ({missing_cols}) AFTER cleaning. Cleaned columns found: {list(df.columns)}. 跳过。")
            return None

        # --- Rest of the function (Keep as is) ---
        df = df.drop(columns=['time'])
        df['annotation_class'] = df['annotation'].apply(map_annotation_to_class)
        df = df.drop(columns=['annotation'])
        df['pid'] = pid

        features_cols = SENSOR_FEATURES_COLS # Use the predefined list

        if df[features_cols].isnull().values.any():
            # print(f"信息: 文件 {file_path} 包含 NaN，执行填充...")
            df[features_cols] = df[features_cols].ffill().bfill()
            if df[features_cols].isnull().values.any():
                print(f"错误: 文件 {file_path} 填充后仍含 NaN。跳过。")
                return None
        return df

    except FileNotFoundError:
        # print(f"错误: 文件 {file_path} 未找到。")
        return None
    except Exception as e:
        # Add more context to the error
        print(f"处理文件 {file_path} 时发生错误 (类型: {type(e).__name__}): {e}，跳过。")
        # You might want to print the traceback for complex errors:
        # import traceback
        # traceback.print_exc()
        return None


# --- Main Execution Logic (Adjust DATA_DIR if necessary, rest is the same) ---
if __name__ == "__main__":
    # 1. 确定要使用的所有文件 (P001-P100)
    all_target_files = []
    print(f"在 '{DATA_DIR}' 中查找 P001-P100 的数据文件...") # Adjusted print
    for i in range(1, 101):
        pid = f"P{i:03d}"
        file_path = os.path.join(DATA_DIR, f"{pid}.csv")
        if os.path.exists(file_path):
            all_target_files.append(file_path)
        # else:
            # print(f"文件 {file_path} 未找到，跳过。")

    if not all_target_files:
        raise ValueError(f"在 '{DATA_DIR}' 目录中未找到任何 P001-P100 的 CSV 文件。")
    print(f"找到 {len(all_target_files)} 个目标数据文件。")

    # 2. 加载所有找到的文件的数据
    print("\n加载所有目标文件的数据...")
    all_dfs = [load_preprocess_add_pid(f) for f in tqdm(all_target_files, desc="加载文件")]
    all_dfs = [df for df in all_dfs if df is not None and not df.empty]
    if not all_dfs:
        raise ValueError("未能成功加载任何有效数据！请检查文件格式、路径和 load_preprocess_add_pid 函数中的错误消息。") # Modified error

    all_df_combined = pd.concat(all_dfs, ignore_index=True)
    print(f"合并后的数据形状: {all_df_combined.shape}")

    # 3. 拟合 Scaler
    print("\n拟合 StandardScaler...")
    scaler = StandardScaler()
    if all_df_combined[SENSOR_FEATURES_COLS].isnull().values.any():
        print("警告: 合并数据中发现 NaN，将使用 ffill/bfill 填充...")
        all_df_combined[SENSOR_FEATURES_COLS] = all_df_combined[SENSOR_FEATURES_COLS].ffill().bfill()
        if all_df_combined[SENSOR_FEATURES_COLS].isnull().values.any():
            raise ValueError("数据填充 NaN 后仍然存在 NaN。无法拟合 Scaler。")
    scaler.fit(all_df_combined[SENSOR_FEATURES_COLS])
    print(f"Scaler 已在以下特征上拟合 (基于 {len(all_dfs)} 个成功加载的文件): {SENSOR_FEATURES_COLS}") # Adjusted count message

    # 4. 拟合 OneHotEncoders (Keep as is)
    all_pids = all_df_combined['pid'].unique()
    sex_encoder = None
    age_encoder = None
    # ... (rest of the encoder fitting logic remains the same) ...
    if PARTICIPANT_METADATA and SEX_TO_CODE:
        print("尝试拟合 Sex Encoder...")
        sex_values_in_meta = set(PARTICIPANT_METADATA[pid]['sex']
                                 for pid in all_pids if pid in PARTICIPANT_METADATA
                                 and 'sex' in PARTICIPANT_METADATA[pid]
                                 and PARTICIPANT_METADATA[pid]['sex'] is not None
                                 and PARTICIPANT_METADATA[pid]['sex'] in SEX_TO_CODE)
        valid_sex_values_for_fit = [[s] for s in sex_values_in_meta]
        if valid_sex_values_for_fit:
            sex_categories = sorted(list(SEX_TO_CODE.keys()), key=lambda k: SEX_TO_CODE[k])
            sex_encoder = OneHotEncoder(categories=[sex_categories], sparse_output=False, handle_unknown='ignore')
            try:
                sex_encoder.fit(valid_sex_values_for_fit)
                print(f"Sex Encoder 已拟合 (类别: {sex_encoder.categories_[0]})。")
            except Exception as e:
                print(f"错误: 拟合 Sex Encoder 失败: {e}。Sex Encoder 将为 None。")
                sex_encoder = None
        else:
            print("警告: 在所有加载文件对应的元数据中未找到有效的性别值。Sex Encoder 将为 None。")
    else:
        print("警告: 缺少 Sex 元数据或 SEX_TO_CODE 映射。Sex Encoder 将为 None。")

    if PARTICIPANT_METADATA and AGE_GROUP_TO_CODE:
        print("尝试拟合 Age Encoder...")
        age_groups_in_meta = set(PARTICIPANT_METADATA[pid]['age_group']
                                 for pid in all_pids if pid in PARTICIPANT_METADATA
                                 and 'age_group' in PARTICIPANT_METADATA[pid]
                                 and PARTICIPANT_METADATA[pid]['age_group'] is not None
                                 and PARTICIPANT_METADATA[pid]['age_group'] in AGE_GROUP_TO_CODE)
        valid_age_groups_for_fit = [[ag] for ag in age_groups_in_meta]
        if valid_age_groups_for_fit:
            age_categories = sorted(list(AGE_GROUP_TO_CODE.keys()), key=lambda k: AGE_GROUP_TO_CODE[k])
            age_encoder = OneHotEncoder(categories=[age_categories], sparse_output=False, handle_unknown='ignore')
            try:
                age_encoder.fit(valid_age_groups_for_fit)
                print(f"Age Encoder 已拟合 (类别: {age_encoder.categories_[0]})。")
            except Exception as e:
                print(f"错误: 拟合 Age Encoder 失败: {e}。Age Encoder 将为 None。")
                age_encoder = None
        else:
            print("警告: 在所有加载文件对应的元数据中未找到有效的年龄组值。Age Encoder 将为 None。")
    else:
        print("警告: 缺少 Age 元数据或 AGE_GROUP_TO_CODE 映射。Age Encoder 将为 None。")

    # 5. 保存拟合的 Scaler 和 Encoders (Keep as is)
    scaler_filename = 'scaler_jerk_all_100.pkl'
    sex_encoder_filename = 'sex_encoder_all_100.pkl'
    age_encoder_filename = 'age_encoder_all_100.pkl'
    print(f"\n保存拟合的预处理器...")
    try:
        with open(scaler_filename, 'wb') as f: pickle.dump(scaler, f)
        print(f"  - StandardScaler 已保存到: {scaler_filename}")
    except Exception as e: print(f"  - 错误: 保存 StandardScaler 失败: {e}")
    if sex_encoder is not None:
        try:
            with open(sex_encoder_filename, 'wb') as f: pickle.dump(sex_encoder, f)
            print(f"  - Sex OneHotEncoder 已保存到: {sex_encoder_filename}")
        except Exception as e: print(f"  - 错误: 保存 Sex Encoder 失败: {e}")
    else:
        if os.path.exists(sex_encoder_filename): os.remove(sex_encoder_filename)
        print(f"  - Sex Encoder 为 None，未保存 {sex_encoder_filename}。")
    if age_encoder is not None:
        try:
            with open(age_encoder_filename, 'wb') as f: pickle.dump(age_encoder, f)
            print(f"  - Age OneHotEncoder 已保存到: {age_encoder_filename}")
        except Exception as e: print(f"  - 错误: 保存 Age Encoder 失败: {e}")
    else:
        if os.path.exists(age_encoder_filename): os.remove(age_encoder_filename)
        print(f"  - Age Encoder 为 None，未保存 {age_encoder_filename}。")

    print("\n脚本执行完毕。")