# 036 added 反常表现域诊断

## 目标

- 针对 `034/035` 暴露出的 `added 21-24` 反常高估，做一轮不重训主模型的诊断；
- 先回答以下问题：
  - `added` 是否改变了训练输入通道口径；
  - `added` 在机制特征空间里更像哪些 `final` 工况；
  - `工况22` 是否是一个单独的异常机制点；
  - `rpm-only` 基线与 `TinyTCN` 外部预测相比差多少。

## 输入与口径

- 主数据：
  - `data/final/dataset_manifest.csv`
  - `data/final/datasets/工况*.csv`
- added 数据：
  - `data/added/dataset_manifest.csv`
  - `data/added/standardized_datasets/工况21-24.csv`
- 清洗逻辑：复用 `src/current.data_loading`
- 机制特征：复用 `src/try/030_case_mechanism_clustering/`
- 既有外部预测结果：
  - `outputs/try/034_added_validation_label_check/added_case_predictions.csv`
  - `outputs/try/035_added_validation_with_full_final_pool/added_case_predictions.csv`

## 运行方式

```powershell
.venv\Scripts\python.exe src/try/036_added_domain_diagnosis/run_added_domain_diagnosis.py
```

## 输出

- 输出目录：`outputs/try/036_added_domain_diagnosis/`
- 固定产物：
  - `added_mechanism_diagnostics.csv`
  - `nearest_final_cases.csv`
  - `added_baseline_comparison.csv`
  - `added_feature_outliers.csv`
  - `case22_reference_comparison.csv`
  - `projection_scatter.png`
  - `case22_spectrum_comparison.png`
  - `summary.md`
