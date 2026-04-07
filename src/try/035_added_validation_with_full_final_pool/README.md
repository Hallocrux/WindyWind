# 035 added 外部验证（包含难工况训练池）

## 目标

- 在 `034` 的基础上做一个最小对照；
- 保持数据口径、模型和窗长不变；
- 只把训练池改成：
  - 使用 `data/final/` 中全部带标签工况
  - 不再去掉 `工况1 / 3 / 17 / 18`
- 再对 `data/added/` 的 `工况21-24` 做外部预测。

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
.venv\Scripts\python.exe src/try/035_added_validation_with_full_final_pool/run_added_validation_with_full_final_pool.py
```

## 输出

- 输出目录：`outputs/try/035_added_validation_with_full_final_pool/`
- 固定产物：
  - `added_case_predictions.csv`
  - `summary.csv`
  - `summary.md`
