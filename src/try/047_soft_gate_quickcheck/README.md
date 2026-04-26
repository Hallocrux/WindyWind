# 047 soft gate quickcheck

## 目标

- 在不使用原始波形的前提下，验证是否可以用 `case-level soft gate` 为两个专家输出连续融合比例；
- 不先做硬二分，而是直接预测：
  - `g in [0, 1]`
- 最终预测写成：
  - `pred = (1 - g) * pred_base + g * pred_enhanced`

## 专家定义

- `base expert`
  - `TinyTCN_multiscale_late_fusion_2s_8s`
- `enhanced expert`
  - `true_rpm -> rpm_knn4 + TinyTCN all_channels midband(3.0-6.0Hz) @ w=0.3`

## 输入

- `outputs/try/026_tinytcn_priority1_quickcheck/full19_multiscale_late_fusion_2s_8s_case_level.csv`
- `outputs/try/035_added_validation_with_full_final_pool/added_case_predictions.csv`
- `outputs/try/044_final_loco_fft_midband_fusion_check/seed_runs/seed_*_case_level_predictions.csv`
- `outputs/try/042_rpm_learned_midband_multiseed_stability_check/seed_case_level_predictions.csv`
- `data/final/dataset_manifest.csv`
- `data/final/datasets/工况{ID}.csv`
- `data/added/dataset_manifest.csv`
- `data/added/standardized_datasets/工况21-24.csv`

## 运行方式

```powershell
uv run python src/try/047_soft_gate_quickcheck/run_soft_gate_quickcheck.py
```

## 输出

- 输出目录：`outputs/try/047_soft_gate_quickcheck/`
- 主要文件：
  - `gate_feature_table.csv`
  - `case_level_predictions.csv`
  - `summary_by_variant.csv`
  - `summary_by_domain.csv`
  - `summary.md`

## 方法说明

- gate 特征只使用 `case-level` 聚合特征；
- gate 监督信号使用每个 case 对应的最优连续融合比例：
  - `g* = clip((y - pred_base) / (pred_enhanced - pred_base), 0, 1)`
- gate 模型比较：
  - `RidgeCV`
  - `HistGradientBoostingRegressor`
- 评估协议：
  - 对 `23` 个带标签工况做统一 case-level `LOOCV`
  - 并分别汇报：
    - `final` 子集
    - `added` 子集
