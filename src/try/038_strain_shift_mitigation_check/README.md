# 038 应变侧漂移缓解快速验证

## 目标

- 在 `037` 已确认 added 高估主要由应变侧输入驱动后，快速验证两类最小修复手段：
  - 只对应变做 `per-case zscore`
  - 只对应变做 `>2Hz high-pass`
- 判断 `all_channels` 是否还能被修复到接近 `acc_only`；
- 如果修不回来，为后续把 added 主线切到 `acc + rpm` 提供证据。

## 输入与口径

- 主数据：
  - `data/final/dataset_manifest.csv`
  - `data/final/datasets/工况*.csv`
- added 数据：
  - `data/added/dataset_manifest.csv`
  - `data/added/standardized_datasets/工况21-24.csv`
- 清洗逻辑：复用 `src/current.data_loading`
- 外部验证模型：
  - `TinyTCN@5s`
- 训练池：
  - `full_final_pool`
  - `clean_final_pool`（去掉 `工况1 / 3 / 17 / 18`）

## 运行方式

```powershell
.venv\Scripts\python.exe src/try/038_strain_shift_mitigation_check/run_strain_shift_mitigation_check.py
```

## 输出

- 输出目录：`outputs/try/038_strain_shift_mitigation_check/`
- 固定产物：
  - `variant_config_table.csv`
  - `case_level_predictions.csv`
  - `summary.csv`
  - `case22_focus.csv`
  - `summary.md`
