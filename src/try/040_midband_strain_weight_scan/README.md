# 040 中频应变细扫与融合权重验证

## 目标

- 在 `039` 已锁定 `3-6Hz` 为当前最可迁移应变频带后，继续细化：
  - `3-6Hz` 邻域的带通窗口是否还能更优；
  - `acc + strain(3-6Hz)` 的融合权重哪个更稳。

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
.venv\Scripts\python.exe src/try/040_midband_strain_weight_scan/run_midband_strain_weight_scan.py
```

## 输出

- 输出目录：`outputs/try/040_midband_strain_weight_scan/`
- 固定产物：
  - `variant_config_table.csv`
  - `case_level_predictions.csv`
  - `summary.csv`
  - `case22_focus.csv`
  - `best_fusion_reference.csv`
  - `summary.md`
