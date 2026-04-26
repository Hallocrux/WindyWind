# 049 TCN soft gate persist

## 目标

- 使用 `TinyTCN` 直接学习连续 gate 比例 `g in [0, 1]`
- gate 输入使用原始多通道时序窗口
- 训练好的模型与归一化统计持久化保存，后续复跑尽量复用 checkpoint

## 专家定义

- `base expert`
  - `TinyTCN_multiscale_late_fusion_2s_8s`
- `enhanced expert`
  - `true_rpm -> rpm_knn4 + TinyTCN all_channels midband(3.0-6.0Hz) @ w=0.3`

## 运行方式

```powershell
uv run python src/try/049_tcn_soft_gate_persist/run_tcn_soft_gate_persist.py
```

## 输出

- 输出目录：`outputs/try/049_tcn_soft_gate_persist/`
- 主要文件：
  - `dataset_table.csv`
  - `case_level_predictions.csv`
  - `summary_by_variant.csv`
  - `summary_by_domain.csv`
  - `models/checkpoints/`
  - `summary.md`
