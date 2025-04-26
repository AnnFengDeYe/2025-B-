# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import IterableDataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import glob
import os
from tqdm import tqdm
import math
from collections import Counter
import matplotlib.pyplot as plt

# --- 从 metadata.py 导入元数据和编码信息 ---
try:
    from ..Metadata.metadata import PARTICIPANT_METADATA, AGE_GROUP_TO_CODE, SEX_TO_CODE, NUM_AGE_GROUPS, NUM_SEX_CODES, NUM_DEMOGRAPHIC_FEATURES
except ImportError:
    print("错误：无法导入 metadata.py。请确保该文件存在且包含必要的变量。")
    print("将使用一些默认值，功能可能受限。")
    PARTICIPANT_METADATA = {'P001': {'age_group': '18-24', 'sex': 'Male'}, 'P002': {'age_group': '25-34', 'sex': 'Female'}}
    AGE_GROUP_TO_CODE = {'18-24': 0, '25-34': 1, '35-49': 2, '50+': 3}
    SEX_TO_CODE = {'Male': 0, 'Female': 1}
    NUM_AGE_GROUPS = len(AGE_GROUP_TO_CODE)
    NUM_SEX_CODES = len(SEX_TO_CODE)
    NUM_DEMOGRAPHIC_FEATURES = NUM_AGE_GROUPS + NUM_SEX_CODES


# --- 配置 ---
DATA_DIR = '../ProcessedData/ProcessedTrainData'        # 计算Jerk和Mag预处理后的训练数据所在目录
VAL_FILES_COUNT = 10
TEST_SET_SIZE = 20


# --- 超参数 ---
WINDOW_SIZE = 10000
STEP = 2500
BATCH_SIZE = 32
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5
NUM_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 10
MIN_DELTA = 0.001
NUM_SENSOR_FEATURES = 7
NUM_TOTAL_FEATURES = None
NUM_CLASSES = 5
CNN_FILTERS = 64
KERNEL_SIZE = 5
LSTM_HIDDEN_SIZE = 128
LSTM_LAYERS = 2
DROPOUT_RATE = 0.3

# --- 设备配置 ---
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    if torch.backends.mps.is_built():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
        print("MPS 后端可用但未构建，回退到 CPU。")
else:
    device = torch.device('cpu')
print(f"使用的设备: {device}")


# --- 数据预处理函数 ---
def map_annotation_to_class(annotation):
    if annotation < 1.0: return 0
    elif 1.0 <= annotation < 1.6: return 1
    elif 1.6 <= annotation < 3.0: return 2
    elif 3.0 <= annotation < 6.0: return 3
    else: return 4

# ... (load_preprocess_add_pid 不变)
def load_preprocess_add_pid(file_path):
    try:
        pid = os.path.basename(file_path).split('.')[0]
        if pid not in PARTICIPANT_METADATA:
            return None
        df = pd.read_csv(file_path)
        required_columns = ['time', 'x', 'y', 'z', 'magnitude', 'annotation', 'Jerk_x', 'Jerk_y', 'Jerk_z']
        if not all(col in df.columns for col in required_columns):
            print(f"警告: 文件 {file_path} 缺少必需列 ({[c for c in required_columns if c not in df.columns]})。跳过。")
            return None
        df = df.drop(columns=['time'])
        df['annotation_class'] = df['annotation'].apply(map_annotation_to_class)
        df = df.drop(columns=['annotation'])
        df['pid'] = pid
        features_cols = ['x', 'y', 'z', 'magnitude', 'Jerk_x', 'Jerk_y', 'Jerk_z']
        if df[features_cols].isnull().values.any():
            df[features_cols] = df[features_cols].ffill().bfill()
            if df[features_cols].isnull().values.any():
                print(f"错误: 文件 {file_path} 填充后仍含 NaN。跳过。")
                return None
        # <<< OPTIMIZATION (Optional but good practice) >>> Convert types early
        # This might slightly reduce memory for the large combined dataframe,
        # although the primary issue is the .unique() call.
        for col in features_cols:
             df[col] = df[col].astype(np.float32)
        df['annotation_class'] = df['annotation_class'].astype(np.int64) # or smaller int if classes allow
        # df['pid'] can remain object/string
        return df
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"处理文件 {file_path} 出错: {e}，跳过。")
        return None


# --- Iterable PyTorch 数据集 (MODIFIED: Accept unique_pids_list) ---
class IterableSensorDataset(IterableDataset):
    # <<< MODIFIED >>> Added unique_pids_list parameter
    def __init__(self, data_df, scaler, window_size, step, metadata, age_encoder, sex_encoder, unique_pids_list):
        super(IterableSensorDataset).__init__()
        self.window_size = window_size
        self.step = step
        self.scaler = scaler
        self.metadata = metadata
        self.age_encoder = age_encoder
        self.sex_encoder = sex_encoder
        self.sensor_features_cols = ['x', 'y', 'z', 'magnitude', 'Jerk_x', 'Jerk_y', 'Jerk_z']

        if data_df.empty:
            print("警告: 传入 IterableSensorDataset 的 DataFrame 为空。")
            self.scaled_sensor_features = np.array([])
            self.labels = np.array([])
            self.pids = np.array([])
            self.num_samples = 0
            self.estimated_num_sequences = 0
            self.encoded_demo_cache = {}
            return

        required_cols = self.sensor_features_cols + ['annotation_class', 'pid']
        if not all(col in data_df.columns for col in required_cols):
             missing_cols = [col for col in required_cols if col not in data_df.columns]
             raise ValueError(f"DataFrame 缺少必需列: {missing_cols}")

        sensor_features_df = data_df[self.sensor_features_cols]
        labels_series = data_df['annotation_class']
        # Store PIDs as NumPy array for faster indexing later in __iter__ if needed
        # Keep it as pandas Series might also be ok, but np array is common
        self.pids = data_df['pid'].values
        self.scaled_sensor_features = self.scaler.transform(sensor_features_df)
        self.labels = labels_series.values
        self.num_samples = len(self.scaled_sensor_features)

        if self.num_samples >= self.window_size:
             self.estimated_num_sequences = max(0, (self.num_samples - self.window_size) // self.step + 1)
        else:
            self.estimated_num_sequences = 0

        # --- MODIFIED ---
        # Use the pre-calculated list of unique PIDs instead of deriving from the large data_df
        if unique_pids_list is None:
             # Fallback, but likely to cause MemoryError again if data_df is huge
             print("警告: 未提供 unique_pids_list，尝试从 DataFrame 计算 (可能导致内存错误)")
             unique_pids_to_process = data_df['pid'].unique()
        else:
             unique_pids_to_process = unique_pids_list # Use the provided list/set

        print(f"为 {len(unique_pids_to_process)} 个唯一 PID 预编码人口统计学特征...")
        self.encoded_demo_cache = {}
        # unique_pids = data_df['pid'].unique() # <<< THIS LINE CAUSED THE ERROR AND IS REMOVED/REPLACED
        for pid in tqdm(unique_pids_to_process, desc="预编码人口统计特征"): # Iterate over the smaller list
            if pid in self.metadata:
                meta = self.metadata[pid]
                sex_val = meta.get('sex')
                age_group_val = meta.get('age_group')
                encoded_features = []
                valid_demo = True
                # ... (编码逻辑保持不变)
                if self.sex_encoder and sex_val is not None:
                    try: encoded_features.append(self.sex_encoder.transform([[sex_val]])[0])
                    except Exception: valid_demo = False; # print(f"Warn: Sex encoding failed for {pid}")
                elif self.sex_encoder is None: pass
                else: valid_demo = False
                if self.age_encoder and age_group_val is not None:
                    try: encoded_features.append(self.age_encoder.transform([[age_group_val]])[0])
                    except Exception: valid_demo = False; # print(f"Warn: Age encoding failed for {pid}")
                elif self.age_encoder is None: pass
                else: valid_demo = False

                if valid_demo and encoded_features:
                    # Ensure encoded features are concatenated correctly and are float32
                    concatenated_features = np.concatenate(encoded_features).astype(np.float32)
                    self.encoded_demo_cache[pid] = concatenated_features
                    # Optional: Check dimension only once
                    # if 'expected_demo_dim' not in locals():
                    #     expected_demo_dim = len(concatenated_features)
                    # elif len(concatenated_features) != expected_demo_dim:
                    #     print(f"Warn: Inconsistent demo feature dimension for {pid}. Expected {expected_demo_dim}, got {len(concatenated_features)}")

                elif valid_demo and not encoded_features:
                     self.encoded_demo_cache[pid] = np.array([], dtype=np.float32) # Case where demo features exist but are empty
            #else:
                 # print(f"Info: PID {pid} not found in metadata for encoding.") # Optional log
        print("人口统计特征预编码完成。")
        # --- END MODIFICATION ---

    def __iter__(self):
        # ... (迭代逻辑不变)
        if self.num_samples < self.window_size:
            return iter([])
        worker_info = torch.utils.data.get_worker_info()
        end_index = max(0, self.num_samples - self.window_size)
        if worker_info is None:
            worker_id = 0; num_workers = 1
        else:
            worker_id = worker_info.id; num_workers = worker_info.num_workers
        first_start_index_for_worker = worker_id * self.step
        for i in range(first_start_index_for_worker, end_index + 1, self.step * num_workers):
            # Optimized index calculation slightly
            window_end_idx = i + self.window_size #- 1 # Index for label and PID
            # Get PID directly from the numpy array using the end index of the window's data
            current_pid = self.pids[window_end_idx -1] # -1 because window_end_idx is exclusive upper bound for slice below

            if current_pid not in self.encoded_demo_cache:
                # This check might be less frequent now due to pre-computation,
                # but still good practice.
                continue # Skip if no demo features found/encoded for this PID

            demo_features = self.encoded_demo_cache[current_pid]

            # Get sensor sequence slice
            sensor_seq = self.scaled_sensor_features[i : window_end_idx] # Slice up to window_end_idx

            # Check sequence length just in case
            if sensor_seq.shape[0] != self.window_size:
                # This should ideally not happen with correct range calculation, but safety check
                # print(f"Warn: Incorrect window size at index {i}. Expected {self.window_size}, got {sensor_seq.shape[0]}. Skipping.")
                continue

            # Get the label corresponding to the end of the window
            label = self.labels[window_end_idx -1] # -1 because label corresponds to last point in window

            # Combine features
            if demo_features.size > 0:
                # Using np.broadcast_to is potentially more memory efficient than np.tile
                # demo_features_repeated = np.tile(demo_features[np.newaxis, :], (self.window_size, 1))
                try:
                    # Ensure demo_features is 1D array before broadcasting
                    if demo_features.ndim != 1:
                        print(f"Warn: demo_features for PID {current_pid} is not 1D ({demo_features.shape}). Skipping.")
                        continue
                    demo_features_repeated = np.broadcast_to(demo_features, (self.window_size, demo_features.shape[0]))
                    combined_seq = np.concatenate((sensor_seq, demo_features_repeated), axis=1)
                except ValueError as e:
                     print(f"Error concatenating features at index {i}, pid {current_pid}: {e}. Sensor shape: {sensor_seq.shape}, Demo shape: {demo_features.shape}, Repeated demo shape: {demo_features_repeated.shape}")
                     continue
            else:
                combined_seq = sensor_seq # Only sensor features

            # Check total features dimension against global variable
            if NUM_TOTAL_FEATURES is not None and combined_seq.shape[1] != NUM_TOTAL_FEATURES:
                # print(f"Warning: Sequence feature dimension mismatch at index {i}. Expected {NUM_TOTAL_FEATURES}, got {combined_seq.shape[1]}. Skipping.")
                continue

            yield torch.tensor(combined_seq, dtype=torch.float32), \
                  torch.tensor(label, dtype=torch.long)


# --- Attention Layer ---
# ... (代码不变)
class Attention(nn.Module):
    def __init__(self, feature_dim):
        super(Attention, self).__init__()
        self.attention_fc = nn.Linear(feature_dim, 1, bias=False)
        self.softmax = nn.Softmax(dim=1)
    def forward(self, lstm_output):
        attention_weights = self.attention_fc(lstm_output)
        attention_weights = attention_weights.squeeze(2)
        attention_weights = self.softmax(attention_weights)
        attention_weights = attention_weights.unsqueeze(2)
        weighted_output = lstm_output * attention_weights
        context_vector = torch.sum(weighted_output, dim=1)
        return context_vector


# --- CNN-LSTM + Attention 模型 ---
# ... (代码不变)
class CNN_LSTM_Attention_Model(nn.Module):
    def __init__(self, input_size, cnn_filters, kernel_size, lstm_hidden_size, num_layers,
                 num_classes, dropout_rate):
        super(CNN_LSTM_Attention_Model, self).__init__()
        if input_size <= 0: raise ValueError(f"模型序列输入维度必须 > 0，当前为 {input_size}")
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=cnn_filters, kernel_size=kernel_size, padding='same')
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        lstm_input_features = cnn_filters
        self.lstm = nn.LSTM(input_size=lstm_input_features, hidden_size=lstm_hidden_size, num_layers=num_layers,
                            batch_first=True, dropout=dropout_rate if num_layers > 1 else 0, bidirectional=True)
        self.attention = Attention(lstm_hidden_size * 2)
        self.dropout = nn.Dropout(dropout_rate)
        fc_input_size = lstm_hidden_size * 2
        self.fc = nn.Linear(fc_input_size, num_classes)

    def forward(self, x):
        if x.shape[2] != self.conv1.in_channels:
            raise ValueError(f"输入序列特征维度 {x.shape[2]} (来自数据加载器) 与 Conv1d 期望通道 {self.conv1.in_channels} (来自模型初始化input_size) 不符。请检查 NUM_TOTAL_FEATURES 计算和模型初始化。")
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        context_vector = self.attention(lstm_out)
        x = self.dropout(context_vector)
        out = self.fc(x)
        return out


# --- 训练和评估函数 ---
# ... (train_model 不变)
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs,
                train_dataset_size_estimate, val_dataset_size_estimate,
                early_stopping_patience, min_delta):
    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []
    lr_history = []
    best_val_accuracy = 0.0
    best_model_state = None
    epochs_no_improve = 0
    best_epoch = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        train_batches = math.ceil(train_dataset_size_estimate / BATCH_SIZE) if train_dataset_size_estimate and BATCH_SIZE > 0 else 0
        if train_batches == 0:
            print(f"警告: 训练集估计大小为 0 或 BATCH_SIZE 无效，无法进行训练周期 {epoch+1}。")
            train_loss_history.append(float('nan'))
            train_acc_history.append(float('nan'))
            val_loss_history.append(float('nan'))
            val_acc_history.append(float('nan'))
            lr_history.append(optimizer.param_groups[0]['lr'])
            continue
        train_pbar = tqdm(train_loader, desc=f"周期 {epoch+1}/{num_epochs} [训练中]", total=train_batches)
        current_lr = optimizer.param_groups[0]['lr']
        lr_history.append(current_lr)
        for sequences, labels in train_pbar:
            sequences, labels = sequences.to(device), labels.to(device)
            optimizer.zero_grad()
            if sequences.dim() != 3 or sequences.shape[0] != labels.shape[0]:
                 print(f"\n错误：批次序列形状 {sequences.shape} 或与标签 {labels.shape} 不匹配！跳过。")
                 continue
            try:
                outputs = model(sequences)
                loss = criterion(outputs, labels)
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"\n警告: 训练损失为 NaN/Inf，跳过反向传播。")
                    continue
                loss.backward()
                optimizer.step()
                batch_size = labels.size(0)
                running_loss += loss.item() * batch_size
                _, predicted = torch.max(outputs.data, 1)
                total_train += batch_size
                correct_train += (predicted == labels).sum().item()
                train_pbar.set_postfix({'损失': f"{loss.item():.4f}",
                                        '准确率': f"{(predicted == labels).float().mean().item():.4f}",
                                        'LR': f"{current_lr:.1e}"})
            except ValueError as ve:
                print(f"\n训练期间模型前向传播出错 (ValueError): {ve} - 跳过此批次")
                continue
            except RuntimeError as re:
                 print(f"\n训练期间模型前向传播出错 (RuntimeError): {re} - 跳过此批次")
                 continue
        epoch_loss = running_loss / total_train if total_train > 0 else float('nan')
        epoch_acc = correct_train / total_train if total_train > 0 else float('nan')
        train_loss_history.append(epoch_loss)
        train_acc_history.append(epoch_acc)
        val_loss, val_acc = float('nan'), float('nan')
        if val_loader is not None and val_dataset_size_estimate > 0:
            val_loss, val_acc = evaluate_model(model, val_loader, criterion, val_dataset_size_estimate,
                                               desc=f"周期 {epoch+1}/{num_epochs} [验证中]",
                                               log_batch_details=False)
            if not math.isnan(val_loss) and not math.isnan(val_acc):
                print(f"\n周期 {epoch+1}/{num_epochs} => "
                      f"训练损失: {epoch_loss:.4f}, 训练准确率: {epoch_acc:.4f} | "
                      f"验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.4f}")
                if scheduler: scheduler.step(val_acc)
                if val_acc > best_val_accuracy + min_delta:
                    best_val_accuracy = val_acc
                    best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    epochs_no_improve = 0
                    best_epoch = epoch + 1
                    print(f"*** 新的最佳验证准确率: {best_val_accuracy:.4f} (周期 {best_epoch}) ***")
                else:
                    epochs_no_improve += 1
                    print(f"验证准确率未提升。连续未提升周期: {epochs_no_improve}/{early_stopping_patience}。")
                if epochs_no_improve >= early_stopping_patience:
                    print(f"\n--- 早停触发 ---")
                    val_loss_history.append(val_loss)
                    val_acc_history.append(val_acc)
                    if best_model_state is not None:
                       model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
                    return model, train_loss_history, train_acc_history, val_loss_history, val_acc_history, lr_history
            else: print(f"\n警告: 验证指标包含 NaN。")
        else: print(f"\n周期 {epoch+1}/{num_epochs} => 训练损失: {epoch_loss:.4f}, 训练准确率: {epoch_acc:.4f} | 验证: 跳过")
        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc)
    print("训练完成 (达到最大周期数或早停)。")
    if best_model_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
        print(f"训练结束。已加载最佳模型状态 (来自周期 {best_epoch}, 验证准确率: {best_val_accuracy:.4f})。")
    elif val_loader is not None: print("警告: 训练完成，但未找到更优模型。")
    return model, train_loss_history, train_acc_history, val_loss_history, val_acc_history, lr_history

# ... (evaluate_model 不变)
def evaluate_model(model, data_loader, criterion, dataset_size_estimate, desc="评估中", log_batch_details=False):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    batch_num = 0
    eval_batches = math.ceil(dataset_size_estimate / BATCH_SIZE) if dataset_size_estimate and BATCH_SIZE > 0 else 0
    if eval_batches == 0:
        print(f"警告: {desc} 数据集大小为 0 或 BATCH_SIZE 无效。跳过评估。")
        return float('nan'), float('nan')
    eval_pbar = tqdm(data_loader, desc=desc, total=eval_batches, leave=False)
    with torch.no_grad():
        for sequences, labels in eval_pbar:
            batch_num += 1
            sequences, labels = sequences.to(device), labels.to(device)
            if sequences.dim() != 3 or sequences.shape[0] != labels.shape[0]:
                 print(f"\n错误 [{desc} - 批次 {batch_num}]: 输入形状不匹配！跳过。Seq: {sequences.shape}, Label: {labels.shape}")
                 continue
            try:
                outputs = model(sequences)
                if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                     print(f"\n警告 [{desc} - 批次 {batch_num}]: 模型输出 NaN/Inf。跳过。")
                     continue
                loss = criterion(outputs, labels)
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"\n警告 [{desc} - 批次 {batch_num}]: 损失 NaN/Inf。跳过。")
                    continue
            except ValueError as ve:
                print(f"\n评估期间模型前向传播出错 (ValueError) [{desc} - 批次 {batch_num}]: {ve} - 跳过此批次")
                continue
            except RuntimeError as re:
                 print(f"\n评估期间模型前向传播出错 (RuntimeError) [{desc} - 批次 {batch_num}]: {re} - 跳过此批次")
                 continue
            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            _, predicted = torch.max(outputs.data, 1)
            total += batch_size
            batch_correct = (predicted == labels).sum().item()
            correct += batch_correct
            batch_acc = batch_correct / batch_size if batch_size > 0 else 0
            if log_batch_details and (batch_acc == 0.0 or batch_acc == 1.0) and batch_size > 0:
                 print(f"\n--- {desc} - 批次 {batch_num} (准确率: {batch_acc:.2f}) ---")
                 print(f"  真实: {labels[:10].cpu().numpy()}")
                 print(f"  预测: {predicted[:10].cpu().numpy()}")
                 print(f"--------------------------------------------------")
            eval_pbar.set_postfix({'损失': f"{loss.item():.4f}", '准确率': f"{batch_acc:.4f}"})
    avg_loss = running_loss / total if total > 0 else float('nan')
    accuracy = correct / total if total > 0 else float('nan')
    eval_pbar.close()
    return avg_loss, accuracy

# --- 获取近似类别权重的函数 ---
# ... (get_approx_class_weights 不变)
def get_approx_class_weights(data_df, num_classes):
    print("计算近似类别权重...")
    if 'annotation_class' not in data_df.columns or data_df.empty:
        print("警告: 无法计算权重。返回权重全为1。")
        return torch.ones(num_classes, dtype=torch.float32).to(device)
    class_counts = data_df['annotation_class'].value_counts().sort_index()
    full_counts = [class_counts.get(i, 0) for i in range(num_classes)]
    total_samples = float(sum(full_counts))
    if total_samples == 0:
         print("警告: 总样本数为0。返回权重全为1。")
         return torch.ones(num_classes, dtype=torch.float32).to(device)
    weights = []
    print("原始数据类别分布:")
    for i in range(num_classes):
        count = full_counts[i]
        percentage = (count / total_samples) * 100 if total_samples > 0 else 0
        print(f"  类别 {i}: {count} ({percentage:.2f}%)")
        if count > 0: weight = total_samples / (num_classes * count)
        else: weight = 1.0 # Assign weight 1 to classes with 0 samples? Or handle differently?
        weights.append(weight)
    # Clamp weights to prevent extremely large values for rare classes if needed
    # max_weight = 10.0
    # weights = [min(w, max_weight) for w in weights]
    print(f"近似类别权重: {[f'{w:.4f}' for w in weights]}")
    return torch.tensor(weights, dtype=torch.float32).to(device)

# --- 绘图函数 ---
# ... (plot_training_history 不变)
def plot_training_history(train_loss, train_acc, val_loss, val_acc, lr_history, save_path=None):
    epochs = range(1, len(train_loss) + 1)
    plt.figure(figsize=(12, 16))
    plt.subplot(3, 1, 1)
    plt.plot(epochs, train_loss, 'bo-', label='Training loss')
    if any(not math.isnan(v) for v in val_loss):
        plt.plot(epochs, val_loss, 'ro--', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs'); plt.ylabel('Loss')
    plt.legend(); plt.grid(True)
    plt.subplot(3, 1, 2)
    plt.plot(epochs, train_acc, 'bo-', label='Training accuracy')
    if any(not math.isnan(v) for v in val_acc):
        plt.plot(epochs, val_acc, 'ro--', label='Validation accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs'); plt.ylabel('Accuracy')
    plt.legend(); plt.grid(True)
    plt.subplot(3, 1, 3)
    plt.plot(epochs, lr_history, 'go-', label='Learning Rate')
    plt.title('Learning Rate per Epoch')
    plt.xlabel('Epochs'); plt.ylabel('Learning Rate')
    is_log_scale = False
    if len(lr_history) > 1 :
        min_lr = min(lr for lr in lr_history if lr > 0) # Avoid zero if exists
        max_lr = max(lr_history)
        if min_lr > 0 and max_lr / min_lr > 10:
             plt.yscale('log')
             plt.ylabel('Learning Rate (log scale)')
             is_log_scale = True
    if not is_log_scale: plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0)) # Use sci notation if not log
    plt.legend(); plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"训练历史图已保存至: {save_path}")
    plt.show()

# --- 主执行逻辑 ---
if __name__ == "__main__":
    # 1. 确定文件路径
    # ... (代码不变)
    all_pids_in_meta = list(PARTICIPANT_METADATA.keys())
    all_available_files = []
    print("检查可用数据文件 (P001-P100)...")
    processed_pids = set()
    for pid in sorted(all_pids_in_meta):
        if pid.startswith('P') and pid[1:].isdigit() and 1 <= int(pid[1:]) <= 100:
             if pid in processed_pids: continue
             file_path = os.path.join(DATA_DIR, f"{pid}.csv")
             if os.path.exists(file_path):
                 all_available_files.append(file_path)
                 processed_pids.add(pid)
    if not all_available_files:
        raise ValueError("未找到任何有效 CSV 文件。请确保文件在 DATA_DIR 指定的目录中。")
    print(f"找到 {len(all_available_files)} 个有效数据文件。")


    # 2. 随机打乱与划分
    # ... (代码不变)
    np.random.seed(42)
    np.random.shuffle(all_available_files)
    num_available = len(all_available_files)
    num_test_files = TEST_SET_SIZE
    num_val_files = VAL_FILES_COUNT
    if num_available < num_val_files + num_test_files + 1:
        print(f"警告: 可用文件 ({num_available}) 不足验证({num_val_files}) + 测试({num_test_files}) + 至少1个训练文件。调整...")
        num_test_files = min(num_test_files, max(0, int(num_available * 0.2)))
        num_val_files = min(num_val_files, max(0, int(num_available * 0.1)))
        num_train_files = num_available - num_test_files - num_val_files
        if num_train_files <= 0: raise ValueError("调整后仍无足够文件用于训练。")
        VAL_FILES_COUNT = num_val_files
        TEST_SET_SIZE = num_test_files
        print(f"调整后: 训练 {num_train_files}, 验证 {VAL_FILES_COUNT}, 测试 {TEST_SET_SIZE}。")
    test_files = all_available_files[:TEST_SET_SIZE]
    val_files = all_available_files[TEST_SET_SIZE : TEST_SET_SIZE + VAL_FILES_COUNT]
    train_files = all_available_files[TEST_SET_SIZE + VAL_FILES_COUNT:]
    if not train_files: raise ValueError("划分后训练文件列表为空！请检查文件数量和划分逻辑。")
    print(f"\n数据集划分:\n  训练: {len(train_files)}, 验证: {len(val_files)}, 测试: {len(test_files)}")


    # 4. 加载训练数据并拟合预处理器
    print("\n加载训练数据以拟合预处理器...")
    train_dfs = [load_preprocess_add_pid(f) for f in tqdm(train_files, desc="加载训练文件")]
    train_dfs = [df for df in train_dfs if df is not None and not df.empty]
    if not train_dfs:
        raise ValueError("无有效训练数据加载！检查 load_preprocess_add_pid 函数和文件格式。")

    # <<< MODIFIED >>> Extract unique PIDs from file list *before* concatenation or heavy processing
    # This avoids the memory error caused by .unique() on the large combined dataframe
    train_pids_from_files = sorted(list(set(os.path.basename(f).split('.')[0] for f in train_files)))
    print(f"从文件名中提取到 {len(train_pids_from_files)} 个唯一的训练集 PID。")
    # Do the same for validation and test sets
    val_pids_from_files = sorted(list(set(os.path.basename(f).split('.')[0] for f in val_files))) if val_files else []
    test_pids_from_files = sorted(list(set(os.path.basename(f).split('.')[0] for f in test_files))) if test_files else []
    print(f"从文件名中提取到 {len(val_pids_from_files)} 个唯一的验证集 PID。")
    print(f"从文件名中提取到 {len(test_pids_from_files)} 个唯一的测试集 PID。")


    print("合并训练数据...")
    # Concatenate AFTER extracting PIDs needed for dataset init
    train_df_combined = pd.concat(train_dfs, ignore_index=True)
    print(f"合并训练数据形状: {train_df_combined.shape}")
    # Explicitly delete the list of individual dataframes to potentially free memory
    del train_dfs
    import gc
    gc.collect() # Suggest garbage collection


    # 5. 拟合 Scaler (在所有7个传感器+Jerk特征上)
    print("拟合 StandardScaler...")
    scaler = StandardScaler()
    sensor_feature_cols = ['x', 'y', 'z', 'magnitude', 'Jerk_x', 'Jerk_y', 'Jerk_z']
    # Check NaN (although load_preprocess_add_pid should handle it, check again)
    if train_df_combined[sensor_feature_cols].isnull().values.any():
        print("警告: 合并训练数据中发现 NaN，将使用 ffill/bfill 填充...")
        # Filling NaNs on a potentially huge dataframe can also be memory intensive
        # Consider if this step is truly necessary or if NaNs were handled properly before
        try:
            train_df_combined[sensor_feature_cols] = train_df_combined[sensor_feature_cols].ffill().bfill()
            if train_df_combined[sensor_feature_cols].isnull().values.any():
                raise ValueError("训练数据填充 NaN 后仍然存在 NaN。处理失败。")
        except MemoryError:
             raise MemoryError("在合并后的训练数据上填充 NaN 时内存不足。检查 load_preprocess_add_pid 中的 NaN 处理。")
    # Fitting scaler might still require significant memory if train_df_combined is huge
    # but usually less than operations like .unique() on object columns.
    try:
        scaler.fit(train_df_combined[sensor_feature_cols])
        print("Scaler 已在传感器+Jerk特征上拟合。")
    except MemoryError:
        # If scaler fitting fails, maybe fit on a sample? Less accurate but might work.
        print("警告: StandardScaler 拟合时内存不足。尝试在数据子集上拟合...")
        sample_size = min(len(train_df_combined), 5_000_000) # Fit on ~5M samples or less
        scaler.fit(train_df_combined[sensor_feature_cols].sample(n=sample_size, random_state=42))
        print(f"Scaler 已在 {sample_size} 个样本的子集上拟合。")


    # 6. 拟合 OneHotEncoders (使用从文件名提取的PID列表)
    # <<< MODIFIED >>> Use train_pids_from_files for fitting encoders
    num_demo_features_actual = 0
    sex_encoder = None
    age_encoder = None

    # Fit encoders based on the metadata corresponding to the PIDs *actually* present in the training files
    pids_for_encoder_fitting = train_pids_from_files # Use the list derived from filenames

    if PARTICIPANT_METADATA and SEX_TO_CODE:
        print("拟合 Sex Encoder...")
        # Filter metadata based on PIDs present in the training files
        sex_values_in_train_meta = set(
            PARTICIPANT_METADATA[pid]['sex']
            for pid in pids_for_encoder_fitting # Use filtered list
            if pid in PARTICIPANT_METADATA and 'sex' in PARTICIPANT_METADATA[pid] and PARTICIPANT_METADATA[pid]['sex'] is not None
        )
        valid_sex_values_for_fit = [[s] for s in sex_values_in_train_meta if s in SEX_TO_CODE]
        if valid_sex_values_for_fit:
            sex_categories = sorted(list(SEX_TO_CODE.keys()), key=lambda k: SEX_TO_CODE[k])
            sex_encoder = OneHotEncoder(categories=[sex_categories], sparse_output=False, handle_unknown='ignore') # ignore unknown might be safer
            try:
                # Fit on the unique valid values found
                unique_valid_sex_values = sorted(list(sex_values_in_train_meta & set(SEX_TO_CODE.keys())))
                if unique_valid_sex_values:
                     sex_encoder.fit([[v] for v in unique_valid_sex_values])
                     print(f"Sex Encoder 已拟合 (类别: {sex_encoder.categories_[0]})。")
                     num_demo_features_actual += len(sex_encoder.categories_[0])
                else:
                     print("警告: 未找到有效的性别值用于拟合 Sex Encoder。")
                     sex_encoder = None # Ensure it's None if fit fails
            except Exception as e: print(f"错误: 拟合 Sex Encoder 失败: {e}。"); sex_encoder = None
        else: print("警告: 训练集元数据中未找到在 SEX_TO_CODE 中定义的有效性别值。"); sex_encoder = None
    else: print("警告: 缺少 Sex 元数据或 SEX_TO_CODE 映射。"); sex_encoder = None

    if PARTICIPANT_METADATA and AGE_GROUP_TO_CODE:
        print("拟合 Age Encoder...")
        # Filter metadata based on PIDs present in the training files
        age_groups_in_train_meta = set(
            PARTICIPANT_METADATA[pid]['age_group']
            for pid in pids_for_encoder_fitting # Use filtered list
            if pid in PARTICIPANT_METADATA and 'age_group' in PARTICIPANT_METADATA[pid] and PARTICIPANT_METADATA[pid]['age_group'] is not None
        )
        valid_age_groups_for_fit = [[ag] for ag in age_groups_in_train_meta if ag in AGE_GROUP_TO_CODE]
        if valid_age_groups_for_fit:
            age_categories = sorted(list(AGE_GROUP_TO_CODE.keys()), key=lambda k: AGE_GROUP_TO_CODE[k])
            age_encoder = OneHotEncoder(categories=[age_categories], sparse_output=False, handle_unknown='ignore') # ignore unknown might be safer
            try:
                 # Fit on the unique valid values found
                unique_valid_age_groups = sorted(list(age_groups_in_train_meta & set(AGE_GROUP_TO_CODE.keys())))
                if unique_valid_age_groups:
                    age_encoder.fit([[v] for v in unique_valid_age_groups])
                    print(f"Age Encoder 已拟合 (类别: {age_encoder.categories_[0]})。")
                    num_demo_features_actual += len(age_encoder.categories_[0])
                else:
                    print("警告: 未找到有效的年龄组值用于拟合 Age Encoder。")
                    age_encoder = None # Ensure it's None if fit fails
            except Exception as e: print(f"错误: 拟合 Age Encoder 失败: {e}。"); age_encoder = None
        else: print("警告: 训练集元数据中未找到在 AGE_GROUP_TO_CODE 中定义的有效年龄组值。"); age_encoder = None
    else: print("警告: 缺少 Age 元数据或 AGE_GROUP_TO_CODE 映射。"); age_encoder = None


    NUM_DEMOGRAPHIC_FEATURES = num_demo_features_actual
    NUM_TOTAL_FEATURES = NUM_SENSOR_FEATURES + NUM_DEMOGRAPHIC_FEATURES
    print(f"最终人口统计学特征数: {NUM_DEMOGRAPHIC_FEATURES}")
    print(f"最终序列总特征数 (传感器[含Jerk] + 人口统计): {NUM_TOTAL_FEATURES}")
    if NUM_TOTAL_FEATURES <= 0:
        raise ValueError("总特征数必须 > 0！")


    # 7. 加载验证和测试数据 (加载过程不变，但后续处理使用提取的PID)
    print("\n加载验证数据...")
    val_dfs = [load_preprocess_add_pid(f) for f in tqdm(val_files, desc="加载验证文件")]
    val_dfs = [df for df in val_dfs if df is not None and not df.empty]
    val_df_combined = pd.concat(val_dfs, ignore_index=True) if val_dfs else pd.DataFrame()
    print(f"合并验证数据形状: {val_df_combined.shape}")
    del val_dfs # Free memory
    gc.collect()

    print("\n加载测试数据...")
    test_dfs = [load_preprocess_add_pid(f) for f in tqdm(test_files, desc="加载测试文件")]
    test_dfs = [df for df in test_dfs if df is not None and not df.empty]
    test_df_combined = pd.concat(test_dfs, ignore_index=True) if test_dfs else pd.DataFrame()
    print(f"合并测试数据形状: {test_df_combined.shape}")
    del test_dfs # Free memory
    gc.collect()


    # 8. 创建 Iterable 数据集
    # <<< MODIFIED >>> Pass the pre-calculated PID lists to the dataset constructor
    if 'age_encoder' not in locals(): age_encoder = None # Ensure they exist even if fitting failed
    if 'sex_encoder' not in locals(): sex_encoder = None

    print("\n创建数据集...")
    train_dataset = IterableSensorDataset(train_df_combined, scaler, WINDOW_SIZE, STEP, PARTICIPANT_METADATA, age_encoder, sex_encoder, unique_pids_list=train_pids_from_files)

    # It's important to free up train_df_combined *if possible*.
    # However, IterableSensorDataset holds a reference to it (or rather, its numpy arrays).
    # If memory is still tight, you might need a more advanced Dataset that reads files on the fly.
    # For now, let's keep train_df_combined as the dataset needs its arrays.
    # print("尝试释放合并的训练 DataFrame 内存...")
    # del train_df_combined
    # gc.collect()

    val_dataset = IterableSensorDataset(val_df_combined, scaler, WINDOW_SIZE, STEP, PARTICIPANT_METADATA, age_encoder, sex_encoder, unique_pids_list=val_pids_from_files) if not val_df_combined.empty else None
    # del val_df_combined # Release memory if val set is large
    # gc.collect()

    test_dataset = IterableSensorDataset(test_df_combined, scaler, WINDOW_SIZE, STEP, PARTICIPANT_METADATA, age_encoder, sex_encoder, unique_pids_list=test_pids_from_files) if not test_df_combined.empty else None
    # del test_df_combined # Release memory if test set is large
    # gc.collect()


    # 9. 创建 DataLoaders
    # ... (代码不变)
    num_data_workers = 0 # Set to 0 for IterableDataset unless using specific worker handling
    print(f"使用 {num_data_workers} 个数据加载工作进程。")
    # persistent_workers not compatible with num_workers=0
    use_persistent = False # num_data_workers > 0 and device.type != 'mps'
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=num_data_workers, pin_memory=(device.type == 'cuda'), persistent_workers=use_persistent) if train_dataset and train_dataset.estimated_num_sequences > 0 else None
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=num_data_workers, pin_memory=(device.type == 'cuda'), persistent_workers=use_persistent) if val_dataset and val_dataset.estimated_num_sequences > 0 else None
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=num_data_workers, pin_memory=(device.type == 'cuda'), persistent_workers=use_persistent) if test_dataset and test_dataset.estimated_num_sequences > 0 else None

    if train_loader is None: raise ValueError("训练 DataLoader 为空。检查数据集大小和估计序列数。")
    if val_loader is None: print("警告: 验证 DataLoader 为空。")
    if test_loader is None: print("警告: 测试 DataLoader 为空。")


    # 10. 初始化模型
    # ... (代码不变)
    print(f"\n初始化模型 (序列输入特征: {NUM_TOTAL_FEATURES})...")
    model = CNN_LSTM_Attention_Model(
        input_size=NUM_TOTAL_FEATURES, cnn_filters=CNN_FILTERS, kernel_size=KERNEL_SIZE,
        lstm_hidden_size=LSTM_HIDDEN_SIZE, num_layers=LSTM_LAYERS,
        num_classes=NUM_CLASSES, dropout_rate=DROPOUT_RATE
    ).to(device)
    print("\n模型架构:")
    # print(model) # Comment out if too verbose
    print(f"总可训练参数: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # 11. 初始化损失和优化器
    # <<< MODIFIED >>> Calculate class weights based on the original DF *before* deleting it
    # Or, if train_df_combined is too large even for this, estimate weights from a sample or file counts
    use_class_weights = True
    if use_class_weights:
        print("计算类别权重...")
        # Option 1: Use the combined DF if it still exists and fits memory for this calc
        # if 'train_df_combined' in locals() and not train_df_combined.empty:
        #    class_weights = get_approx_class_weights(train_df_combined, NUM_CLASSES)
        # Option 2: Estimate from annotations in metadata or sample files (more complex)
        # For now, assume train_df_combined is available here or was small enough to keep
        # This might need adjustment if train_df_combined was deleted aggressively
        if 'train_df_combined' in locals() and train_df_combined is not None and not train_df_combined.empty:
             class_weights = get_approx_class_weights(train_df_combined, NUM_CLASSES)
             criterion = nn.CrossEntropyLoss(weight=class_weights)
             print(f"使用加权损失 (权重设备: {class_weights.device})。")
             # Now we can potentially delete train_df_combined if not needed by Dataset internals later
             # print("尝试释放合并的训练 DataFrame 内存 (计算完权重后)...")
             # del train_df_combined
             # gc.collect()
        else:
             print("警告: train_df_combined 不可用，无法计算类别权重。使用标准损失。")
             criterion = nn.CrossEntropyLoss()
             print("使用标准损失 (未加权)。")

    else:
        criterion = nn.CrossEntropyLoss()
        print("使用标准损失 (未加权)。")

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    print(f"使用 AdamW 优化器 (LR={LEARNING_RATE}, WD={WEIGHT_DECAY})。")

    # 12. 初始化学习率调度器
    scheduler = None
    if val_loader is not None:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=5, min_lr=1e-6)
        print(f"使用 ReduceLROnPlateau LR 调度器 (基于验证准确率, factor=0.2, patience=5)。")
    else:
        print("无验证集，不使用 LR 调度器。")


    # 13. 训练模型
    print("\n--- 开始训练 ---")
    train_estimate = train_dataset.estimated_num_sequences if train_dataset else 0
    val_estimate = val_dataset.estimated_num_sequences if val_dataset else 0
    if 'scheduler' not in locals(): scheduler = None

    trained_model, train_loss_hist, train_acc_hist, val_loss_hist, val_acc_hist, lr_hist = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler, NUM_EPOCHS,
        train_estimate, val_estimate,
        EARLY_STOPPING_PATIENCE, MIN_DELTA
    )

    # 14. 绘图
    history_plot_path = f'training_history_Attention_Jerk_ws{WINDOW_SIZE}_st{STEP}_bs{BATCH_SIZE}_cf{CNN_FILTERS}_ks{KERNEL_SIZE}_lh{LSTM_HIDDEN_SIZE}_L{LSTM_LAYERS}.png'
    if train_loss_hist:
        plot_training_history(train_loss_hist, train_acc_hist, val_loss_hist, val_acc_hist, lr_hist, save_path=history_plot_path)
    else:
        print("训练未执行或未完成，跳过绘制训练历史图。")


    # 15. 在测试集上评估
    print("\n--- 在测试集上评估 ---")
    test_loss, test_acc = float('nan'), float('nan')
    if test_loader is not None:
        test_estimate = test_dataset.estimated_num_sequences if test_dataset else 0
        if test_estimate > 0:
            test_loss, test_acc = evaluate_model(trained_model, test_loader, criterion,
                                                 test_estimate, desc="测试中", log_batch_details=False)
            if not math.isnan(test_loss) and not math.isnan(test_acc):
                print(f"\n最终测试结果:\n  损失: {test_loss:.4f}\n  准确率: {test_acc:.4f}")
            else: print("测试评估返回 NaN 或 Inf。")
        else: print("测试数据集估计大小为 0，跳过评估。")
    else: print("测试 DataLoader 为空，跳过最终评估。")

    # 16. 保存模型
    final_model_path = f'final_model_Attention_Jerk_ws{WINDOW_SIZE}_st{STEP}_bs{BATCH_SIZE}_cf{CNN_FILTERS}_ks{KERNEL_SIZE}_lh{LSTM_HIDDEN_SIZE}_L{LSTM_LAYERS}.pth'
    try:
        torch.save(trained_model.state_dict(), final_model_path)
        print(f"最终模型状态已保存至: {final_model_path}")
    except Exception as e:
        print(f"错误: 保存模型失败: {e}")


    print("\n脚本执行完毕。")
