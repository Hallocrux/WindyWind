# 037 工况22 标签链路与模态外部对照

## 目标

- 对 `工况22` 做一轮更明确的标签链路审计；
- 在与 `035` 一致的 `5s` 外部验证口径下，比较：
  - `all channels`
  - `strain-only`
  - `acc-only`
  - `rpm-only`
- 判断 `added 21-24` 的系统性高估主要由哪类输入驱动。

## 输入与口径

- 主数据：
  - `data/final/dataset_manifest.csv`
  - `data/final/datasets/工况*.csv`
- added 数据：
  - `data/added/dataset_manifest.csv`
  - `data/added/datasets/*.csv`
  - `data/added/standardized_datasets/工况21-24.csv`
- 清洗逻辑：复用 `src/current.data_loading`
- 外部验证模型：
  - `TinyTCN@5s`
- 训练池：
  - `full_final_pool`
  - `clean_final_pool`（去掉 `工况1 / 3 / 17 / 18`）

## 运行方式

```powershell
.venv\Scripts\python.exe src/try/037_case22_label_and_modality_check/run_case22_label_and_modality_check.py
```

## 输出

- 输出目录：`outputs/try/037_case22_label_and_modality_check/`
- 固定产物：
  - `file_copy_audit.csv`
  - `label_chain_audit.csv`
  - `modality_case_predictions.csv`
  - `modality_summary.csv`
  - `case22_modality_focus.csv`
  - `summary.md`
