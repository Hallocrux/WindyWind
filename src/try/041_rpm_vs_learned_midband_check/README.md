# 041 解析基线与 Learned 中频分支复核

## 目标

- 对 added 方向做一轮更接近决策层的对照：
  - `rpm_knn4` 解析基线
  - `strain(3.0-6.0Hz)` learned 分支
  - `all_channels + strain(3.0-6.0Hz)` learned 分支
  - 轻量 `Ridge` 版本
  - `rpm + learned` 晚融合
- 判断当前 added 主线更适合：
  - 纯解析
  - 纯 learned
  - 解析 + learned 混合

## 输入与口径

- 主数据：
  - `data/final/dataset_manifest.csv`
  - `data/final/datasets/工况*.csv`
- added 数据：
  - `data/added/dataset_manifest.csv`
  - `data/added/standardized_datasets/工况21-24.csv`
- 清洗逻辑：复用 `src/current.data_loading`
- 训练池：
  - `full_final_pool`
- learned 分支频带：
  - `strain bandpass 3.0-6.0Hz`

## 运行方式

```powershell
.venv\Scripts\python.exe src/try/041_rpm_vs_learned_midband_check/run_rpm_vs_learned_midband_check.py
```

## 输出

- 输出目录：`outputs/try/041_rpm_vs_learned_midband_check/`
- 固定产物：
  - `variant_config_table.csv`
  - `case_level_predictions.csv`
  - `summary.csv`
  - `case22_focus.csv`
  - `decision_reference.csv`
  - `summary.md`
