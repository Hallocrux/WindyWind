# 039 应变可迁移频带筛选

## 目标

- 在 `038` 已证明应变高通可部分修复 added 外推后，继续检查：
  - 是否存在比 `>2Hz high-pass` 更稳定的应变中频带；
  - 该频带是否能接近或超过 `acc_only`；
  - 应变更适合直接并入主模型，还是只作为与 `acc_only` 的辅助晚融合。

## 输入与口径

- 主数据：
  - `data/final/dataset_manifest.csv`
  - `data/final/datasets/工况*.csv`
- added 数据：
  - `data/added/dataset_manifest.csv`
  - `data/added/standardized_datasets/工况21-24.csv`
- 清洗逻辑：复用 `src/current.data_loading`
- 模型：
  - `TinyTCN@5s`
- 训练池：
  - `full_final_pool`

## 运行方式

```powershell
.venv\Scripts\python.exe src/try/039_strain_transfer_band_scan/run_strain_transfer_band_scan.py
```

## 输出

- 输出目录：`outputs/try/039_strain_transfer_band_scan/`
- 固定产物：
  - `variant_config_table.csv`
  - `case_level_predictions.csv`
  - `summary.csv`
  - `case22_focus.csv`
  - `best_band_reference.csv`
  - `summary.md`
