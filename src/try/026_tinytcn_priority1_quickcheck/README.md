# 026 TinyTCN 第一优先级快速验证

## 目标

- 快速验证第一优先级中的两个方向是否有继续投入价值：
  - `1` 多尺度 `TinyTCN`
  - `3` 按工况均衡加权训练
- 避免扩成大矩阵，只针对已经确认更值得看的目标工况：
  - `工况1`
  - `工况3`
  - `工况17`
  - `工况18`

## 验证策略

### 多尺度方向

- 不先实现复杂的新网络结构；
- 先复用 `014` 已有单窗长结果，构造一个最小可检验代理：
  - `TinyTCN@2s`
  - `TinyTCN@8s`
  - 工况级晚融合：`0.5 * pred_2s + 0.5 * pred_8s`
- 如果这个晚融合代理都没有信号，就不值得马上写更复杂的多尺度网络。

### 工况均衡加权方向

- 复用 `5s TinyTCN` 主线；
- 只改训练损失：
  - 每个训练窗口的权重按其所属工况窗口数的倒数设置；
  - 使每个工况在总 loss 中的贡献更接近相同。

## 输入与口径

- 数据入口：
  - `data/final/dataset_manifest.csv`
  - `data/final/datasets/工况*.csv`
- 清洗逻辑：复用 `src/current.data_loading`
- 原始窗口构造：复用 `src/try/012_phase3_end_to_end_shortlist/`
- 基线模型逻辑：复用 `src/try/013_phase3_cnn_tcn_smoke/`
- 已有窗长结果来源：
  - `outputs/try/014_phase3_tcn_window_length_scan/tcn_window_scan_case_level_predictions.csv`

## 运行方式

```powershell
.venv\Scripts\python.exe src/try/026_tinytcn_priority1_quickcheck/run_tinytcn_priority1_quickcheck.py
```

## 输出

- 输出目录：`outputs/try/026_tinytcn_priority1_quickcheck/`
- 固定产物：
  - `variant_case_level_comparison.csv`
  - `variant_summary.csv`
  - `balanced_training_case_window_counts.csv`
  - `summary.md`
