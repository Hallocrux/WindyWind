# Baseline TinyTCN

这个目录固化 `2026-04-05` 当前最强的第三阶段候选 baseline：

- 模型：`TinyTCN`
- 数据口径：
  - `data/final/dataset_manifest.csv`
  - `data/final/datasets/工况*.csv`
- 清洗口径：复用 `src/current.data_loading`
- 切窗口径：
  - `50Hz`
  - `5s` 窗长
  - `2.5s` 步长
- 评估方式：
  - 按工况 `Leave-One-Condition-Out`

## 运行方式

```powershell
uv run python -m src.Baseline_TinyTCN
```

## 输出

- `outputs/Baseline_TinyTCN/model_summary.csv`
- `outputs/Baseline_TinyTCN/case_level_predictions.csv`
- `outputs/Baseline_TinyTCN/unlabeled_predictions.csv`
