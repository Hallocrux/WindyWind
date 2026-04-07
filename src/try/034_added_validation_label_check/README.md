# 034 added 外部验证与可疑标签检查

## 目标

- 使用 `data/added/` 里的新补充工况做外部验证；
- 训练时去掉当前难工况：
  - `工况1`
  - `工况3`
  - `工况17`
  - `工况18`
- 先验证这个“clean pool”对 `added` 数据是否仍然有预测力；
- 再用同一训练池检查可疑工况的标签偏离。

## 输入与口径

- 训练数据：
  - `data/final/dataset_manifest.csv`
  - `data/final/datasets/工况*.csv`
- 外部验证数据：
  - `data/added/dataset_manifest.csv`
  - `data/added/standardized_datasets/工况21-24.csv`
- 清洗逻辑：复用 `src/current.data_loading`
- 模型：
  - `TinyTCN@2s`
  - `TinyTCN@5s`
  - `TinyTCN@8s`
  - `2s+8s` 晚融合

## 运行方式

```powershell
.venv\Scripts\python.exe src/try/034_added_validation_label_check/run_added_validation_label_check.py
```

## 输出

- 输出目录：`outputs/try/034_added_validation_label_check/`
- 固定产物：
  - `added_case_predictions.csv`
  - `suspicious_case_predictions.csv`
  - `summary.csv`
  - `summary.md`
