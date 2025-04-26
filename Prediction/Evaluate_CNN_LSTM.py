# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import glob
import os
from tqdm import tqdm
import math
import pickle
import gc # Garbage collector

# --- 从 metadata.py 导入训练时使用的元数据信息 (主要用于结构和常量) ---
# We need the original constants to define the model structure correctly.
try:
    from ..Metadata.metadata import AGE_GROUP_TO_CODE, SEX_TO_CODE, NUM_AGE_GROUPS, NUM_SEX_CODES
    print("从 metadata.py 加载训练时的编码信息。")
except ImportError:
    print("警告：无法从 metadata.py 加载训练编码信息。将使用默认值。")
    # Provide defaults matching the training script if metadata.py is missing
    AGE_GROUP_TO_CODE = {'18-24': 0, '25-34': 1, '35-49': 2, '50+': 3}
    SEX_TO_CODE = {'Male': 0, 'Female': 1}
    NUM_AGE_GROUPS = len(AGE_GROUP_TO_CODE)
    NUM_SEX_CODES = len(SEX_TO_CODE)

# --- 从 metadata_test.py 导入测试参与者的元数据 ---
try:
    # Assuming the test metadata file uses the same variable names
    from ..Metadata.metadata_test import PARTICIPANT_METADATA as PARTICIPANT_METADATA_TEST
    print("从 metadata_test.py 加载测试参与者元数据。")
except ImportError:
    print("错误：无法导入 metadata_test.py。请确保该文件存在于同一目录中。")
    PARTICIPANT_METADATA_TEST = {} # Use empty dict if file not found

# --- 配置 ---
MODEL_PATH = '../TrainedModelPara/final_model_Attention_Jerk_ws10000_st2500_bs32_cf64_ks5_lh128_L2_7098.pth' # 模型权重文件
SCALER_PATH = '../FitScaler/scaler_jerk_all_100.pkl'         # Scaler 文件
AGE_ENCODER_PATH = '../FitScaler/age_encoder_all_100.pkl'   # Age Encoder 文件
SEX_ENCODER_PATH = '../FitScaler/sex_encoder_all_100.pkl'   # Sex Encoder 文件

TEST_DATA_DIR = '../ProcessedData/ProcessedTestData'                  # 测试数据 P101-P120 预处理后的数据所在的目录
OUTPUT_DIR = '../PredictionOutput'         # 输出预测结果的目录
TEST_PIDS = [f"P{i:03d}" for i in range(101, 121)] # P101 到 P120

# --- 超参数 (必须与训练时一致) ---
WINDOW_SIZE = 10000
# STEP is not directly used for generating inference windows in this chunk-based approach,
# but WINDOW_SIZE is crucial.
NUM_SENSOR_FEATURES = 7 # x, y, z, magnitude, Jerk_x, Jerk_y, Jerk_z
NUM_CLASSES = 5         # 0, 1, 2, 3, 4 (与训练时一致)
CNN_FILTERS = 64
KERNEL_SIZE = 5
LSTM_HIDDEN_SIZE = 128
LSTM_LAYERS = 2
DROPOUT_RATE = 0.3       # 虽然 dropout 在 eval 模式下不生效，但定义模型需要
INFERENCE_BATCH_SIZE = 256 # 可以调整以平衡速度和内存使用
CHUNK_SIZE = 500000      # 处理大文件的块大小，可根据内存调整

# --- 设备配置 ---
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
print(f"使用的设备: {device}")

# --- 模型定义 (需要与训练脚本中的完全一致) ---
class Attention(nn.Module):
    def __init__(self, feature_dim):
        super(Attention, self).__init__()
        self.attention_fc = nn.Linear(feature_dim, 1, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, lstm_output):
        # lstm_output shape: (batch, seq_len, hidden_size * 2)
        attention_weights = self.attention_fc(lstm_output) # (batch, seq_len, 1)
        attention_weights = attention_weights.squeeze(2) # (batch, seq_len)
        attention_weights = self.softmax(attention_weights) # (batch, seq_len)
        attention_weights = attention_weights.unsqueeze(2) # (batch, seq_len, 1)
        weighted_output = lstm_output * attention_weights # (batch, seq_len, hidden_size * 2)
        context_vector = torch.sum(weighted_output, dim=1) # (batch, hidden_size * 2)
        return context_vector

class CNN_LSTM_Attention_Model(nn.Module):
    def __init__(self, input_size, cnn_filters, kernel_size, lstm_hidden_size, num_layers,
                 num_classes, dropout_rate):
        super(CNN_LSTM_Attention_Model, self).__init__()
        if input_size <= 0:
             raise ValueError(f"模型序列输入维度必须 > 0，当前为 {input_size}")
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=cnn_filters, kernel_size=kernel_size, padding='same')
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        # LSTM input features are the output channels of the CNN
        lstm_input_features = cnn_filters
        self.lstm = nn.LSTM(input_size=lstm_input_features, hidden_size=lstm_hidden_size, num_layers=num_layers,
                            batch_first=True, dropout=dropout_rate if num_layers > 1 else 0, bidirectional=True)
        self.attention = Attention(lstm_hidden_size * 2) # Input to attention is output of bidirectional LSTM
        self.dropout = nn.Dropout(dropout_rate)
        # Input to the final FC layer is the output of the attention mechanism
        fc_input_size = lstm_hidden_size * 2
        self.fc = nn.Linear(fc_input_size, num_classes)

    def forward(self, x):
        # Input x shape: (batch_size, sequence_length, num_total_features)
        # Permute for Conv1d: (batch_size, num_total_features, sequence_length)
        if x.dim() != 3:
             raise ValueError(f"Expected 3D input (batch, seq_len, features), but got {x.dim()}D shape: {x.shape}")
        if x.shape[2] != self.conv1.in_channels:
            raise ValueError(f"Input feature dimension mismatch. Expected {self.conv1.in_channels}, got {x.shape[2]}.")

        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x) # Shape: (batch_size, cnn_filters, sequence_length / 2)
        # Permute for LSTM: (batch_size, sequence_length / 2, cnn_filters)
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x) # lstm_out shape: (batch_size, sequence_length / 2, lstm_hidden_size * 2)
        context_vector = self.attention(lstm_out) # context_vector shape: (batch_size, lstm_hidden_size * 2)
        x = self.dropout(context_vector)
        out = self.fc(x) # out shape: (batch_size, num_classes)
        return out

# --- 辅助函数 ---
def load_preprocessors(scaler_path, age_encoder_path, sex_encoder_path):
    """Loads the scaler and encoders from pickle files."""
    print("加载预处理器...")
    try:
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        print(f"  Scaler loaded from {scaler_path}")
    except FileNotFoundError:
        print(f"错误: Scaler 文件未找到: {scaler_path}")
        raise
    except Exception as e:
        print(f"加载 Scaler 出错: {e}")
        raise

    age_encoder = None
    if os.path.exists(age_encoder_path):
        try:
            with open(age_encoder_path, 'rb') as f:
                age_encoder = pickle.load(f)
            print(f"  Age Encoder loaded from {age_encoder_path}")
            print(f"    Age categories: {age_encoder.categories_[0] if hasattr(age_encoder, 'categories_') else 'N/A'}")
        except FileNotFoundError:
            print(f"警告: Age Encoder 文件未找到: {age_encoder_path} (将不使用年龄特征)")
        except Exception as e:
            print(f"加载 Age Encoder 出错: {e}")
            age_encoder = None # Ensure it's None if loading fails
    else:
         print(f"警告: Age Encoder 文件路径不存在: {age_encoder_path} (将不使用年龄特征)")


    sex_encoder = None
    if os.path.exists(sex_encoder_path):
        try:
            with open(sex_encoder_path, 'rb') as f:
                sex_encoder = pickle.load(f)
            print(f"  Sex Encoder loaded from {sex_encoder_path}")
            print(f"    Sex categories: {sex_encoder.categories_[0] if hasattr(sex_encoder, 'categories_') else 'N/A'}")
        except FileNotFoundError:
             print(f"警告: Sex Encoder 文件未找到: {sex_encoder_path} (将不使用性别特征)")
        except Exception as e:
            print(f"加载 Sex Encoder 出错: {e}")
            sex_encoder = None # Ensure it's None if loading fails
    else:
        print(f"警告: Sex Encoder 文件路径不存在: {sex_encoder_path} (将不使用性别特征)")


    return scaler, age_encoder, sex_encoder

def get_demographic_features(pid, metadata, age_encoder, sex_encoder):
    """Gets encoded demographic features for a participant."""
    if pid not in metadata:
        # print(f"警告: 元数据中未找到 PID {pid}。无法添加人口统计学特征。")
        return np.array([], dtype=np.float32)

    meta = metadata[pid]
    encoded_features = []
    valid_demo = True

    sex_val = meta.get('sex')
    if sex_encoder and sex_val is not None:
        try:
            encoded = sex_encoder.transform([[sex_val]])[0]
            encoded_features.append(encoded)
        except Exception as e:
            # This might happen if the sex value is not in the encoder's categories
            # and handle_unknown='error' (though we used 'ignore' during training setup)
            # print(f"警告: 无法编码 PID {pid} 的性别 '{sex_val}': {e}") # Optional warning
            # Append zeros matching the expected output shape if encoding fails but encoder exists
            num_sex_features = len(sex_encoder.categories_[0])
            encoded_features.append(np.zeros(num_sex_features, dtype=np.float32))

    elif sex_encoder is None:
        pass # No sex encoder loaded, do nothing
    # else: # sex_val is None
        # If encoder exists but value is missing, maybe append zeros? Or handle as missing.
        # Let's be consistent: if no value, no feature. If encoding fails, add zeros.

    age_group_val = meta.get('age_group')
    if age_encoder and age_group_val is not None:
         try:
            encoded = age_encoder.transform([[age_group_val]])[0]
            encoded_features.append(encoded)
         except Exception as e:
            # print(f"警告: 无法编码 PID {pid} 的年龄组 '{age_group_val}': {e}") # Optional warning
            num_age_features = len(age_encoder.categories_[0])
            encoded_features.append(np.zeros(num_age_features, dtype=np.float32))

    elif age_encoder is None:
         pass
    # else: # age_group_val is None
        # pass # No value, no feature

    if encoded_features:
        return np.concatenate(encoded_features).astype(np.float32)
    else:
        return np.array([], dtype=np.float32)


# --- 主推理逻辑 ---
if __name__ == "__main__":
    # 1. 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"输出目录 '{OUTPUT_DIR}' 已确认/创建。")

    # 2. 加载预处理器
    try:
        scaler, age_encoder, sex_encoder = load_preprocessors(SCALER_PATH, AGE_ENCODER_PATH, SEX_ENCODER_PATH)
    except Exception as e:
        print(f"无法加载必要的预处理器，脚本终止。错误: {e}")
        exit(1)

    # 3. 确定实际的人口统计学特征数量
    num_demo_features_actual = 0
    if sex_encoder:
        try:
             num_demo_features_actual += len(sex_encoder.categories_[0])
        except AttributeError:
             print("警告: 加载的 Sex Encoder 缺少 'categories_' 属性。")
    if age_encoder:
        try:
             num_demo_features_actual += len(age_encoder.categories_[0])
        except AttributeError:
             print("警告: 加载的 Age Encoder 缺少 'categories_' 属性。")

    NUM_TOTAL_FEATURES = NUM_SENSOR_FEATURES + num_demo_features_actual
    print(f"传感器特征数: {NUM_SENSOR_FEATURES}")
    print(f"实际加载的人口统计学特征数: {num_demo_features_actual}")
    print(f"模型期望的总输入特征数: {NUM_TOTAL_FEATURES}")

    if NUM_TOTAL_FEATURES <= 0:
        print("错误: 总特征数计算结果为 0 或负数，无法继续。")
        exit(1)

    # 4. 加载模型
    print(f"\n加载模型架构和权重从 {MODEL_PATH}...")
    model = CNN_LSTM_Attention_Model(
        input_size=NUM_TOTAL_FEATURES, # 使用计算得到的总特征数
        cnn_filters=CNN_FILTERS,
        kernel_size=KERNEL_SIZE,
        lstm_hidden_size=LSTM_HIDDEN_SIZE,
        num_layers=LSTM_LAYERS,
        num_classes=NUM_CLASSES,
        dropout_rate=DROPOUT_RATE # Dropout is inactive in eval mode anyway
    )
    try:
        # Load state dict; ensure map_location handles CPU/GPU loading
        map_location = device if torch.cuda.is_available() or device.type == 'mps' else torch.device('cpu')
        model.load_state_dict(torch.load(MODEL_PATH, map_location=map_location))
        model.to(device) # Move model to the target device
        model.eval()     # 设置为评估模式 (禁用 dropout, batchnorm 使用运行统计数据等)
        print("模型加载成功并设置为评估模式。")
        # print("\n模型架构:")
        # print(model) # Optional: Print model structure
    except FileNotFoundError:
        print(f"错误: 模型文件未找到: {MODEL_PATH}")
        exit(1)
    except Exception as e:
        print(f"加载模型权重出错: {e}")
        # Often happens if the architecture definition doesn't match the saved weights
        print("请确认模型定义与训练时完全一致，并且 NUM_TOTAL_FEATURES 计算正确。")
        exit(1)

    # 5. 查找测试文件
    test_files = []
    print(f"\n在 '{TEST_DATA_DIR}' 中查找测试文件 (P101-P120)...")
    for pid in TEST_PIDS:
        file_path = os.path.join(TEST_DATA_DIR, f"{pid}.csv")
        if os.path.exists(file_path):
            test_files.append(file_path)
        else:
            print(f"警告: 未找到测试文件 {file_path}")
    if not test_files:
        print("错误: 未找到任何测试文件。请检查 TEST_DATA_DIR 和文件名。")
        exit(1)
    print(f"找到 {len(test_files)} 个测试文件。")

    # 6. 逐个文件进行推理
    sensor_features_cols = ['x', 'y', 'z', 'magnitude', 'Jerk_x', 'Jerk_y', 'Jerk_z']
    required_cols = ['time'] + sensor_features_cols # 需要从原始文件中读取的列

    # --- Buffer for overlapping windows between chunks ---
    overlap_buffer = {} # Dictionary to store buffer per file {pid: pd.DataFrame}

    for file_path in test_files:
        pid = os.path.basename(file_path).split('.')[0]
        output_file_path = os.path.join(OUTPUT_DIR, f"{pid}.csv")
        print(f"\n--- 开始处理文件: {file_path} (PID: {pid}) ---")

        # 获取该参与者的人口统计学特征 (只需获取一次)
        demo_features = get_demographic_features(pid, PARTICIPANT_METADATA_TEST, age_encoder, sex_encoder)
        if demo_features.size == 0:
             print(f"  警告: 未找到 PID {pid} 的人口统计学特征或未加载编码器。模型将仅使用传感器特征。")
             current_total_features = NUM_SENSOR_FEATURES
        else:
             current_total_features = NUM_SENSOR_FEATURES + len(demo_features)
             print(f"  已获取 PID {pid} 的人口统计学特征 ({len(demo_features)} 个)。")

        # 检查特征维度是否与模型期望匹配
        if current_total_features != NUM_TOTAL_FEATURES:
             print(f"  严重警告: PID {pid} 计算得到的特征数 ({current_total_features}) 与模型期望的 ({NUM_TOTAL_FEATURES}) 不符！")
             print(f"  这可能是因为该 PID 的元数据缺失或与训练数据中的编码器不兼容。")
             print(f"  跳过文件 {pid} 以避免模型出错。")
             continue # Skip this file

        file_results = [] # Store results (time, x, y, z, prediction) for the current file
        first_chunk = True
        total_rows_processed = 0
        buffer_df = pd.DataFrame() # Buffer for overlapping data between chunks

        try:
            # 使用 chunksize 读取大文件
            # Use iterator=True and low_memory=False for potentially better memory handling with chunks
            chunk_iter = pd.read_csv(file_path, chunksize=CHUNK_SIZE, iterator=True, low_memory=False)

            print(f"  开始逐块读取和处理 (块大小: {CHUNK_SIZE})...")
            for i, chunk in enumerate(tqdm(chunk_iter, desc=f"  处理 {pid} 块", unit="块")):
                # print(f"    处理块 {i+1}...") # Verbose logging if needed

                # --- 合并 Buffer ---
                if not buffer_df.empty:
                    # print(f"      合并前块 ({len(buffer_df)} 行) 到当前块 ({len(chunk)} 行)")
                    chunk = pd.concat([buffer_df, chunk], ignore_index=True)
                    # print(f"      合并后大小: {len(chunk)} 行")

                # --- 检查必需列 ---
                if not all(col in chunk.columns for col in required_cols):
                    missing = [col for col in required_cols if col not in chunk.columns]
                    print(f"    错误: 块 {i+1} 缺少必需列: {missing}。跳过此文件。")
                    file_results = [] # Clear any partial results
                    break # Stop processing this file

                # --- 提取所需列和处理 NaN ---
                current_data = chunk[required_cols].copy()
                sensor_data = current_data[sensor_features_cols]

                if sensor_data.isnull().values.any():
                    # print(f"    块 {i+1}: 发现 NaN，执行 ffill/bfill...")
                    sensor_data = sensor_data.ffill().bfill()
                    # Check again, especially for NaNs at the very beginning
                    if sensor_data.isnull().values.any():
                         print(f"    错误: 块 {i+1} 填充后仍含 NaN。可能文件开头全是 NaN。跳过此文件。")
                         file_results = []
                         break
                # Ensure data types are correct for scaler
                sensor_data = sensor_data.astype(np.float64)

                # --- 缩放传感器特征 ---
                scaled_sensor_features = scaler.transform(sensor_data)

                # --- 合并特征 ---
                if demo_features.size > 0:
                    # Tile demographic features to match the number of rows in the chunk
                    demo_features_repeated = np.tile(demo_features, (len(scaled_sensor_features), 1))
                    combined_features = np.concatenate((scaled_sensor_features, demo_features_repeated), axis=1)
                else:
                    combined_features = scaled_sensor_features

                # 检查合并后的维度是否正确
                if combined_features.shape[1] != NUM_TOTAL_FEATURES:
                     print(f"    错误: 块 {i+1} 合并后特征维度 ({combined_features.shape[1]}) 与模型期望 ({NUM_TOTAL_FEATURES}) 不符。跳过文件。")
                     file_results = []
                     break

                # --- 生成滑动窗口 ---
                # We need enough data to form at least one window
                if len(combined_features) < WINDOW_SIZE:
                    # print(f"    块 {i+1} 数据不足 ({len(combined_features)} 行)，无法生成窗口。将其设为下一块的 Buffer。")
                    buffer_df = chunk # Keep the original chunk data for the buffer
                    continue # Process next chunk

                num_windows = len(combined_features) - WINDOW_SIZE + 1
                # print(f"    块 {i+1}: 生成 {num_windows} 个窗口...")

                # Use stride_tricks for efficient window generation if memory allows,
                # otherwise, a loop might be necessary for extremely large chunks/windows.
                # Let's use a loop with batching for clarity and reasonable memory use.

                chunk_predictions = np.full(len(chunk), -1, dtype=int) # Initialize predictions for the chunk with -1
                sequences_batch = []
                window_end_indices = [] # Store the original index corresponding to the end of each window

                for k in range(num_windows):
                    window = combined_features[k : k + WINDOW_SIZE]
                    sequences_batch.append(window)
                    window_end_indices.append(k + WINDOW_SIZE - 1) # Index in the 'chunk' dataframe

                    # Process batch when full or at the end of windows
                    if len(sequences_batch) == INFERENCE_BATCH_SIZE or k == num_windows - 1:
                        sequences_tensor = torch.tensor(np.array(sequences_batch), dtype=torch.float32).to(device)

                        with torch.no_grad():
                            try:
                                outputs = model(sequences_tensor)
                                _, predicted_classes = torch.max(outputs.data, 1)
                                batch_preds = predicted_classes.cpu().numpy()
                            except Exception as model_err:
                                print(f"\n    错误: 模型推理失败 (块 {i+1}, 窗口起始 {k - len(sequences_batch) + 1}): {model_err}")
                                print(f"      输入形状: {sequences_tensor.shape}")
                                # Handle error, e.g., skip batch or file
                                batch_preds = np.full(len(sequences_batch), -2, dtype=int) # Use -2 for error marker


                        # Assign predictions back to the correct row index in the chunk
                        for pred_idx, original_idx in enumerate(window_end_indices):
                            # The prediction from window k to k+W-1 applies to row k+W-1
                            chunk_predictions[original_idx] = batch_preds[pred_idx]

                        # Clear batch
                        sequences_batch = []
                        window_end_indices = []

                # --- 准备输出 ---
                # Add the predictions to the original data for this chunk
                current_data['annotation_class'] = chunk_predictions
                # Keep only the rows that received a valid prediction (i.e., where index >= WINDOW_SIZE - 1)
                # And exclude the buffer region for the *next* chunk
                valid_predictions_mask = current_data['annotation_class'] != -1

                # --- 处理 Buffer 和输出 ---
                # Decide which rows to output from this chunk's processing cycle.
                # Output rows from the start of the *original* chunk up to the point
                # needed for the next buffer.

                # Rows to potentially output: indices from buffer_df size up to end of current data
                start_output_idx = len(buffer_df) if not buffer_df.empty else 0
                # Don't output the last WINDOW_SIZE - 1 rows yet, keep them for the next buffer
                end_output_idx = len(current_data) - (WINDOW_SIZE - 1)

                # Ensure end index is not negative and not smaller than start index
                end_output_idx = max(start_output_idx, end_output_idx)

                # Select the rows from current_data (which includes buffer) that correspond to this chunk's *original* data
                # and have valid predictions, excluding the next buffer zone.
                output_df_chunk = current_data.iloc[start_output_idx:end_output_idx].copy()

                # Filter out rows that didn't get a prediction (shouldn't happen in this range if W > 0)
                # but good practice to check. Also filter out error markers (-2).
                output_df_chunk = output_df_chunk[output_df_chunk['annotation_class'] >= 0]


                # --- 追加到文件 ---
                output_cols = ['time', 'x', 'y', 'z', 'annotation_class']
                if not output_df_chunk.empty:
                    if first_chunk:
                        output_df_chunk[output_cols].to_csv(output_file_path, index=False, mode='w', header=True)
                        first_chunk = False
                        print(f"  结果头已写入: {output_file_path}")
                    else:
                        output_df_chunk[output_cols].to_csv(output_file_path, index=False, mode='a', header=False)
                    # print(f"    块 {i+1}: 追加了 {len(output_df_chunk)} 行预测结果。")
                    total_rows_processed += len(output_df_chunk)


                # --- 更新 Buffer ---
                # The new buffer is the last WINDOW_SIZE - 1 rows of the *original* chunk data
                if len(chunk) >= WINDOW_SIZE -1 :
                     buffer_df = chunk.iloc[-(WINDOW_SIZE - 1):].copy()
                     # print(f"    更新 Buffer 为最后 {len(buffer_df)} 行。")
                else:
                     # The current chunk itself becomes the buffer if it's too small
                     buffer_df = chunk.copy()
                     # print(f"    整个块 ({len(buffer_df)} 行) 成为 Buffer。")


                # --- 清理内存 ---
                del current_data, sensor_data, scaled_sensor_features, combined_features, chunk_predictions
                del sequences_tensor, outputs, predicted_classes # Explicitly delete tensors
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                elif device.type == 'mps':
                     # MPS doesn't have an explicit empty_cache, rely on Python GC
                     pass
                gc.collect() # Force garbage collection

            # --- 处理最后一个 Buffer ---
            # After the loop, process the final remaining buffer
            print("  处理最后一个 Buffer...")
            if not buffer_df.empty and len(buffer_df) >= WINDOW_SIZE:
                 current_data = buffer_df[required_cols].copy() # Use the final buffer
                 sensor_data = current_data[sensor_features_cols]
                 # NaN check (less likely here, but good practice)
                 if sensor_data.isnull().values.any():
                     sensor_data = sensor_data.ffill().bfill()
                 if sensor_data.isnull().values.any():
                     print(f"    错误: 最终 Buffer 填充后仍含 NaN。")
                 else:
                     sensor_data = sensor_data.astype(np.float64)
                     scaled_sensor_features = scaler.transform(sensor_data)
                     if demo_features.size > 0:
                         demo_features_repeated = np.tile(demo_features, (len(scaled_sensor_features), 1))
                         combined_features = np.concatenate((scaled_sensor_features, demo_features_repeated), axis=1)
                     else:
                         combined_features = scaled_sensor_features

                     if combined_features.shape[1] == NUM_TOTAL_FEATURES:
                         num_windows = len(combined_features) - WINDOW_SIZE + 1
                         final_predictions = np.full(len(buffer_df), -1, dtype=int)
                         sequences_batch = []
                         window_end_indices = []

                         for k in range(num_windows):
                            window = combined_features[k : k + WINDOW_SIZE]
                            sequences_batch.append(window)
                            window_end_indices.append(k + WINDOW_SIZE - 1)

                            if len(sequences_batch) == INFERENCE_BATCH_SIZE or k == num_windows - 1:
                                sequences_tensor = torch.tensor(np.array(sequences_batch), dtype=torch.float32).to(device)
                                with torch.no_grad():
                                     try:
                                         outputs = model(sequences_tensor)
                                         _, predicted_classes = torch.max(outputs.data, 1)
                                         batch_preds = predicted_classes.cpu().numpy()
                                     except Exception as model_err:
                                         print(f"\n    错误: 模型推理失败 (最终 Buffer): {model_err}")
                                         batch_preds = np.full(len(sequences_batch), -2, dtype=int)

                                for pred_idx, original_idx in enumerate(window_end_indices):
                                     final_predictions[original_idx] = batch_preds[pred_idx]

                                sequences_batch = []
                                window_end_indices = []

                         # Prepare output for the final buffer
                         current_data['annotation_class'] = final_predictions
                         # Output all valid predictions from the final buffer
                         output_df_final = current_data[current_data['annotation_class'] >= 0].copy()

                         # Append to file
                         output_cols = ['time', 'x', 'y', 'z', 'annotation_class']
                         if not output_df_final.empty:
                             if first_chunk: # Should be False now, but handle edge case of only one chunk total
                                 output_df_final[output_cols].to_csv(output_file_path, index=False, mode='w', header=True)
                                 print(f"  结果头已写入: {output_file_path}")
                             else:
                                 output_df_final[output_cols].to_csv(output_file_path, index=False, mode='a', header=False)
                             # print(f"    最终 Buffer: 追加了 {len(output_df_final)} 行预测结果。")
                             total_rows_processed += len(output_df_final)

                         del sequences_tensor, outputs, predicted_classes
                         if device.type == 'cuda': torch.cuda.empty_cache()
                         gc.collect()
                     else:
                          print(f"    错误: 最终 Buffer 特征维度 ({combined_features.shape[1]}) 不匹配模型 ({NUM_TOTAL_FEATURES})。")

            elif not buffer_df.empty:
                 print(f"  最终 Buffer 数据不足 ({len(buffer_df)} 行)，无法生成完整窗口，不进行处理。")


        except FileNotFoundError:
            print(f"错误: 读取文件时未找到: {file_path}")
            continue # Skip to next file
        except Exception as e:
            print(f"处理文件 {file_path} 时发生未预料的错误: {e}")
            import traceback
            traceback.print_exc() # Print detailed traceback for debugging
            continue # Skip to next file

        # --- 清理当前文件的 Buffer ---
        buffer_df = pd.DataFrame() # Reset buffer for the next file
        gc.collect()

        # --- 打印结果样本 ---
        if os.path.exists(output_file_path) and total_rows_processed > 0:
            print(f"  文件 {pid} 处理完成。总共处理并输出了 {total_rows_processed} 行有效预测。")
            try:
                # 读取输出文件的一小部分来展示样本
                output_sample_df = pd.read_csv(output_file_path)
                num_output_rows = len(output_sample_df)
                print(f"  预测结果样本 (来自 {output_file_path}):")

                if num_output_rows > 0:
                    print("    前 10 行预测:")
                    print(output_sample_df.head(10).to_string())
                else:
                    print("    输出文件为空。")

                if num_output_rows > 20:
                    mid_point = num_output_rows // 2
                    print("\n    中间 10 行预测 (大约):")
                    print(output_sample_df.iloc[mid_point-5 : mid_point+5].to_string())

                    print("\n    末尾 10 行预测:")
                    print(output_sample_df.tail(10).to_string())
                elif num_output_rows > 10:
                    print("\n    末尾 {num_output_rows-10} 行预测:")
                    print(output_sample_df.tail(max(0, num_output_rows-10)).to_string())

            except Exception as e:
                print(f"  读取输出文件 {output_file_path} 以显示样本时出错: {e}")
        elif not os.path.exists(output_file_path):
             print(f"  处理完成，但未生成输出文件: {output_file_path} (可能由于错误或无有效数据)")
        else: # Exists but total_rows_processed is 0
             print(f"  处理完成，但未输出有效预测行到: {output_file_path}")


    print("\n--- 所有测试文件处理完毕 ---")