# 050 TCN conservative gate persist

## 目标

- 使用 `TinyTCN` 做更保守的两阶段 gate
- 第一阶段判断是否允许增强
- 第二阶段在 `{0.3, 0.5, 1.0}` 中选择增强权重
- 训练好的模型与归一化统计持久化保存，后续复跑尽量复用 checkpoint

## 专家定义

- `base expert`
  - `TinyTCN_multiscale_late_fusion_2s_8s`
- `enhanced expert`
  - `true_rpm -> rpm_knn4 + TinyTCN all_channels midband(3.0-6.0Hz) @ w=0.3`

## 运行方式

```powershell
uv run python src/try/050_tcn_conservative_gate_persist/run_tcn_conservative_gate_persist.py
```

## 输出

- 输出目录：`outputs/try/050_tcn_conservative_gate_persist/`
- 主要文件：
  - `dataset_table.csv`
  - `case_level_predictions.csv`
  - `summary_by_variant.csv`
  - `summary_by_domain.csv`
  - `models/checkpoints/`
  - `summary.md`
