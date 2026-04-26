import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.optim as optim
import glob

# ==========================================
# 1. 配置参数
# ==========================================
DATA_ROOT = r"C:\Users\Administrator\Desktop\新风电项目\all切片"  # 你的工况文件夹所在的根目录
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001
WINDOW_SIZE = 100  # 必须与你切片时的大小一致
SENSOR_COLS = 20   
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. 数据集加载类
# ==========================================
class WindFarmDataset(Dataset):
    def __init__(self, data_root):
        self.data_list = []   # 存储 .npy 文件的完整路径
        self.labels_list = [] # 存储对应的 [wind_speed, rpm]
        self.folder_path = data_root # <--- 修复1：保存根目录，虽然下面主要用完整路径
        
        # 遍历所有工况文件夹
        folders = [f for f in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, f))]
        
        print(f"发现 {len(folders)} 个工况文件夹，正在加载索引...")
        
        for folder in folders:
            folder_path = os.path.join(data_root, folder)
            label_file = os.path.join(folder_path, "labels.csv")
            
            if not os.path.exists(label_file):
                continue
                
            # 读取标签文件
            labels_df = pd.read_csv(label_file)
            
            for _, row in labels_df.iterrows():
                # 确保读取的列名正确（根据你的报错历史，这里假设CSV里是 'npy_file'）
                npy_filename = row['npy_file'] 
                npy_path = os.path.join(folder_path, npy_filename)
                
                if os.path.exists(npy_path):
                    self.data_list.append(npy_path) # 存完整路径
                    
                    # 存标签，确保列名对应
                    self.labels_list.append([row['wind_speed'], row['rpm']])

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        # 1. 获取文件路径和标签
        npy_path = self.data_list[idx] # <--- 修复2：直接从 list 取路径
        wind_speed, rpm = self.labels_list[idx] # <--- 修复3：直接从 labels_list 取标签

        # 2. 加载数据
        data = np.load(npy_path)

        # --- 维度检查与修复 ---
        if data.shape[1] != SENSOR_COLS:
            # 这里保留你之前的修复逻辑
            if data.shape[1] == 20 and SENSOR_COLS == 10:
                 data = data[:, :10] # 举例：如果数据是20通道但模型要10通道，截取前10个
            # ... 其他修复逻辑 ...

        # 3. 转换为 Tensor (通道, 长度)
        data = torch.tensor(data, dtype=torch.float32).permute(1, 0)

        # 4. 标签转 Tensor
        labels = torch.tensor([wind_speed, rpm], dtype=torch.float32)

        return data, labels # <--- 返回数据 和 合并后的标签

# ==========================================
# 3. 多任务模型定义 (CNN)
# ==========================================
class MultiTaskCNN(nn.Module):
    def __init__(self):
        super(MultiTaskCNN, self).__init__()
        
        # 特征提取层 (处理 10个传感器通道)
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(in_channels=SENSOR_COLS, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2), # 100 -> 50
            
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2), # 50 -> 25
            
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(10) # 强制压缩到长度10
        )
        
        # 展平后的维度: 256 * 10 = 2560
        self.flatten_dim = 256 * 10
        
        # 任务1: 预测风速和转速 (回归)
        self.regressor = nn.Sequential(
            nn.Linear(self.flatten_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2) # 输出: 风速, 转速
        )
        
        # 任务2: 重构传感器数据 (解码器)
        # 这里简化为全连接层重构，实际可用ConvTranspose1d
        self.reconstructor = nn.Sequential(
            nn.Linear(self.flatten_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 100 * SENSOR_COLS), # 输出: 100 * 10
            nn.Sigmoid() # 假设数据已归一化到0-1，否则去掉
        )

    def forward(self, x):
        # x shape: (Batch, 10, 100)
        features = self.feature_extractor(x)
        features_flat = features.view(-1, self.flatten_dim)
        
        # 分支1：预测工况
        pred_labels = self.regressor(features_flat)
        
        # 分支2：重构输入
        recon_data = self.reconstructor(features_flat)
        recon_data = recon_data.view(-1, SENSOR_COLS, WINDOW_SIZE)
        
        return pred_labels, recon_data
class MultiTaskLSTM(nn.Module):
    def __init__(self, input_size=20, hidden_size=128, num_layers=2, dropout=0.3):
        super(MultiTaskLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 1. LSTM 特征提取层
        # batch_first=True: 输入形状为 (Batch, Seq_Len, Features) -> (N, 100, 20)
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout,
            bidirectional=True  # 双向LSTM，效果更好
        )
        
        # 双向LSTM输出维度翻倍 (hidden_size * 2)
        lstm_out_dim = hidden_size * 2
        
        # 2. 任务一：回归预测 (风速 + 转速)
        # 我们取 LSTM 最后一步的输出进行预测
        self.regressor = nn.Sequential(
            nn.Linear(lstm_out_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)  # 输出2个值：风速、转速
        )
        
        # 3. 任务二：数据重构 (Autoencoder部分)
        # 将特征映射回原始数据维度 (100个时间步 * 20个传感器)
        self.reconstructor = nn.Sequential(
            nn.Linear(lstm_out_dim, 100 * 20),
            nn.ReLU(),
        )

    def forward(self, x):
        # x 的形状: (Batch, 20, 100) -> 需要转置为 (Batch, 100, 20) 以适配 LSTM
        x = x.permute(0, 2, 1) 
        
        # LSTM 前向传播
        # h_n: (num_layers * 2, Batch, hidden_size)
        output, (h_n, c_n) = self.lstm(x)
        
        # 获取最后一层的双向特征
        # 拼接 正向最后一层 和 反向最后一层 的隐藏状态
        # h_n shape: (num_layers * 2, batch, hidden)
        # 我们取第 -2 (正向最后一层) 和 -1 (反向最后一层)
        feat_forward = h_n[-2, :, :] 
        feat_backward = h_n[-1, :, :]
        features = torch.cat((feat_forward, feat_backward), dim=1) # (Batch, hidden*2)
        
        # 分支一：预测工况
        pred_labels = self.regressor(features)
        
        # 分支二：重构数据
        # 先通过全连接层，再变回序列形状
        recon_flat = self.reconstructor(features)
        recon_data = recon_flat.view(-1, 100, 20) # (Batch, 100, 20)
        recon_data = recon_data.permute(0, 2, 1)  # 变回 (Batch, 20, 100)
        
        return pred_labels, recon_data
# ==========================================
# 4. 训练过程
# ==========================================
def train():
    # 1. 准备数据
    dataset = WindFarmDataset(DATA_ROOT)
    
    # 简单的归一化 scaler (实际工程中建议保存scaler)
    # 这里为了演示，假设数据已经在 -1 到 1 之间，或者我们做一个简单的 batch norm
    
    # 划分训练/测试 (简单起见，这里全部用于训练演示)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = MultiTaskLSTM().to(DEVICE)
    
    # 损失函数
    criterion_reg = nn.MSELoss()      # 用于风速/转速
    criterion_rec = nn.MSELoss()      # 用于传感器重构
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for data, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            data = data.to(DEVICE)
            labels = labels.to(DEVICE)
            
            optimizer.zero_grad()
            
            # 前向传播
            pred_labels, recon_data = model(data)
            
            # 计算损失
            # 1. 工况预测损失
            loss_reg = criterion_reg(pred_labels, labels)
            
            # 2. 数据重构损失 (让模型学会还原输入)
            loss_rec = criterion_rec(recon_data, data)
            
            # 总损失 (加权)
            loss = loss_reg + 0.5 * loss_rec 
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")
        
    # 保存模型
    torch.save(model.state_dict(), "wind_model.pth")
    print("模型已保存: wind_model.pth")

# ==========================================
# 5. 推理与应用 (解决你的两个问题)
# ==========================================
def inference_demo():
    """
    纯推理演示系统
    只读取包含 .npy 切片的文件夹，不再直接处理 CSV
    """
    # 1. 加载模型
    if not os.path.exists("wind_model.pth"):
        print("❌ 错误：未找到 wind_model.pth，请先运行训练！")
        return

    model = MultiTaskLSTM().to(DEVICE)
    model.load_state_dict(torch.load("wind_model.pth", map_location=DEVICE))
    model.eval()

    print(f"\n✅ 模型加载成功 (设备: {DEVICE})")

    while True:
        print("\n" + "="*50)
        print("           🚀 风电数据推理系统")
        print("="*50)
        print("1. [任务一] 批量预测风速和转速")
        print("2. [任务二] 批量补全缺失传感器数据")
        print("0. 退出程序")
        print("-"*50)

        choice = input("请输入选项 (0-2): ").strip()

        if choice == '0':
            print("👋 程序已退出。")
            break

        # 获取文件夹路径 (两个任务都需要)
        folder_path = input("📁 请输入包含 .npy 切片的文件夹路径: ").strip()
        
        if not os.path.exists(folder_path):
            print(f"❌ 错误：找不到文件夹 '{folder_path}'")
            continue

        # 寻找所有 npy 文件
        npy_files = glob.glob(os.path.join(folder_path, "*.npy"))
        # 过滤，只取 segment 开头的文件
        npy_files = [f for f in npy_files if "segment" in os.path.basename(f).lower()]
        
        if not npy_files:
            print(f"❌ 错误：文件夹 '{folder_path}' 中没有找到 .npy 文件。")
            continue
            
        print(f"🔍 发现 {len(npy_files)} 个切片，开始处理...")

        # ---------------------------------------------------------
        # 选项 1：批量预测风速和转速
        # ---------------------------------------------------------
        if choice == '1':
            all_results = []
            
            for npy_path in sorted(npy_files):
                try:
                    # 加载数据 (100, 20)
                    data_2d = np.load(npy_path)
                    
                    # 检查维度
                    if data_2d.shape[1] != 20:
                        continue
                        
                    # 转置并增加 Batch 维度 -> (1, 20, 100)
                    input_tensor = torch.from_numpy(data_2d.T).unsqueeze(0).float().to(DEVICE)
                    
                    with torch.no_grad():
                        pred_labels, _ = model(input_tensor)
                        wind, rpm = pred_labels.cpu().numpy()[0]
                        
                    filename = os.path.basename(npy_path)
                    print(f"   ✅ [{filename}] 风速: {wind:.2f} m/s | 转速: {rpm:.2f} RPM")
                    all_results.append((filename, wind, rpm))
                    
                except Exception as e:
                    print(f"   ❌ 处理 {npy_path} 失败: {e}")
            
            # 打印统计结果
            if all_results:
                avg_wind = np.mean([r[1] for r in all_results])
                avg_rpm = np.mean([r[2] for r in all_results])
                print(f"\n📈 整体估算 (平均值): 风速 {avg_wind:.2f} m/s | 转速 {avg_rpm:.2f} RPM")

        # ---------------------------------------------------------
        # 选项 2：批量补全缺失数据
        # ---------------------------------------------------------
        elif choice == '2':
            all_completed_data = []
            
            for npy_path in sorted(npy_files):
                try:
                    # 加载残缺数据 (100, 20)
                    incomplete_data = np.load(npy_path)
                    
                    if incomplete_data.shape[1] != 20:
                        continue
                        
                    # 转置并增加 Batch 维度 -> (1, 20, 100)
                    input_tensor = torch.from_numpy(incomplete_data.T).unsqueeze(0).float().to(DEVICE)
                    
                    with torch.no_grad():
                        # 只需要重构数据 (第二个返回值)
                        _, completed_data = model(input_tensor)
                        
                        # 转回 Numpy (100, 20)
                        result_segment = completed_data.cpu().numpy()[0].T
                        
                        all_completed_data.append(result_segment)
                        
                except Exception as e:
                    print(f"   ❌ 处理 {npy_path} 失败: {e}")
            
            # 保存结果
            if all_completed_data:
                # 将所有切片拼接起来 (简单拼接，不考虑重叠平滑)
                final_data = np.concatenate(all_completed_data, axis=0)
                
                # 生成输出文件名
                folder_name = os.path.basename(folder_path)
                out_name = f"{folder_name}_最终修复结果.csv"
                
                # 保存 CSV
                cols = [f"Sensor_{i}" for i in range(SENSOR_COLS)]
                pd.DataFrame(final_data, columns=cols).to_csv(out_name, index=False)
                
                print(f"\n✅ 补全完成！共处理 {len(all_completed_data)} 个切片。")
                print(f"   结果已保存至: {out_name}")
                print(f"   数据预览 (前3行): \n{final_data[:3]}")
if __name__ == "__main__":
    # 1. 先训练
    #train()
    
    # 2. 再推理
    inference_demo()