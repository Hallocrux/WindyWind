import os
import glob
import pandas as pd
import numpy as np

# ================= 配置区域 =================
WINDOW_SIZE = 100   # 滑动窗口长度
STEP_SIZE = 50      # 滑动步长
SENSOR_COLS = 20    # 目标列数 (15个加速度 + 5个应变)
# ===========================================

def process_csv_intelligent(csv_path):
    """
    智能处理函数：
    1. 自动检测数据完整性
    2. 执行特殊规则：WSMS00006 -> WSMS00005
    3. 自动决定输出文件夹名称 (_预测切片 或 _补全切片)
    """
    print(f"\n🛠️ 正在分析: {os.path.basename(csv_path)}")
    
    try:
        # 1. 读取 CSV
        df = pd.read_csv(csv_path)
        
        # 2. 剔除时间列 (假设第一列是时间)
        if df.shape[1] > SENSOR_COLS:
            df = df.iloc[:, 1:]
        
        # 3. 强制转数值 (非数字变 NaN)
        # 注意：这里先不填 0，因为我们要先检查有没有数据
        df = df.apply(pd.to_numeric, errors='coerce')
        
        # 4. 【核心逻辑】处理 WSMS00005 和 WSMS00006 的映射
        # 规则：无论 00005 是否存在，只要有 00006，就用 00006 的数据填充 00005 的位置
        # 假设列名格式为 "WSMSxxxxx.AccX"
        
        # 定义需要处理的轴向
        axes = ['AccX', 'AccY', 'AccZ']
        
        for axis in axes:
            col_00005 = f"WSMS00005.{axis}"
            col_00006 = f"WSMS00006.{axis}"
            
            # 检查 00006 是否存在
            if col_00006 in df.columns:
                # 如果 00006 存在，强制把它的数据覆盖到 00005 列
                # 如果 00005 列不存在，先创建它
                if col_00005 not in df.columns:
                    df[col_00005] = np.nan
                
                # 将 00006 的数据赋值给 00005 (如果 00006 也是 NaN，那填进去也是 NaN)
                df[col_00005] = df[col_00006]
                
                # (可选) 如果你想用 00006 替换 00005 后，把 00006 删掉或清空，可以在这里处理
                # 但通常保留原样也没关系，只要 00005 位置有数就行
        
        # 5. 重新排列列顺序 (确保顺序固定，符合模型训练时的顺序)
        # 假设模型训练时的顺序是：00001-00005 的 X/Y/Z，然后是 Strain 1-5
        target_cols = []
        for i in range(1, 6):
            sensor_id = f"WSMS0000{i}"
            for axis in ['AccX', 'AccY', 'AccZ']:
                target_cols.append(f"{sensor_id}.{axis}")
        
        # 添加应变传感器 (假设列名是 "应变传感器1.chdata" 等)
        for i in range(1, 6):
            target_cols.append(f"应变传感器{i}.chdata")
            
        # 筛选并排序 DataFrame
        # 使用 .get 方法防止列不存在报错，不存在的列会被忽略
        existing_cols = [col for col in target_cols if col in df.columns]
        df = df[existing_cols]
        
        # 6. 填补缺失值
        # 如果列数不够 20，或者某些行有 NaN，统一填 0
        # 这一步决定了它是"预测任务"还是"补全任务"
        is_incomplete = False
        
        # 检查列数是否够
        if len(existing_cols) < SENSOR_COLS:
            print(f"   ⚠️ 检测到缺失传感器通道 (期望{SENSOR_COLS}列，实际{len(existing_cols)}列)")
            is_incomplete = True
            # 补齐列名，防止后续报错 (填0)
            for col in target_cols:
                if col not in df.columns:
                    df[col] = 0
            
            # 重新按目标顺序排列
            df = df[target_cols]
        
        # 检查是否有空行 (NaN)
        if df.isna().any().any():
            print(f"   ⚠️ 检测到数据中有空值 (NaN)，将统一填 0")
            is_incomplete = True
            df = df.fillna(0)
            
        # 转为 numpy
        data_array = df.values.astype(np.float32)
        
        # 7. 确定输出文件夹名称
        base_name = os.path.splitext(os.path.basename(csv_path))[0]
        if is_incomplete:
            output_folder = os.path.join(os.path.dirname(csv_path), f"{base_name}_补全切片")
            print(f"   🎯 判定结果: [补全任务] -> {output_folder}")
        else:
            output_folder = os.path.join(os.path.dirname(csv_path), f"{base_name}_预测切片")
            print(f"   🎯 判定结果: [预测任务] -> {output_folder}")
            
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            
        # 8. 滑动窗口切片
        count = 0
        for start in range(0, len(data_array) - WINDOW_SIZE + 1, STEP_SIZE):
            segment = data_array[start : start + WINDOW_SIZE]
            
            # 最终检查维度
            if segment.shape[1] != SENSOR_COLS:
                 print(f"   ❌ 错误：切片维度 {segment.shape[1]} 不匹配")
                 continue
                 
            npy_filename = f"segment_{count:04d}.npy"
            np.save(os.path.join(output_folder, npy_filename), segment)
            count += 1
            
        print(f"   ✅ 生成 {count} 个切片")
        
    except Exception as e:
        print(f"   ❌ 处理失败: {e}")

# ================= 主程序入口 =================
if __name__ == "__main__":
    print("="*50)
    print("   智能数据预处理工具 (自动判别版)")
    print("="*50)
    
    input_dir = "待处理数据"
    
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)
        print(f"\n📁 已创建文件夹 '{input_dir}'。")
        print(f"请将 CSV 文件放入该文件夹，然后重新运行。")
    else:
        csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
        
        if not csv_files:
            print(f"\n⚠️ 在 '{input_dir}' 中未找到 CSV 文件。")
        else:
            print(f"\n🔍 发现 {len(csv_files)} 个文件，开始智能处理...")
            for csv_file in sorted(csv_files):
                process_csv_intelligent(csv_file)
            
            print("\n🎉 所有任务结束！")