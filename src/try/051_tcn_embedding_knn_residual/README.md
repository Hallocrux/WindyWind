# 051 TinyTCN embedding kNN residual quickcheck

## 目标

- 验证是否可以把 `TinyTCN` 从“直接回归器”改成“表征编码器”；
- 在统一 `final + added` 带标签池上做 case-level `LOOCV`；
- 比较以下路线：
  - `rpm_knn4`
  - `TinyTCN@2s direct`
  - `TinyTCN embedding -> case kNN`
  - `rpm_knn4 + TinyTCN embedding residual kNN`

## 输入与口径

- 主训练域：
  - `data/final/dataset_manifest.csv`
  - `data/final/datasets/工况*.csv`
- 补充域：
  - `data/added/dataset_manifest.csv`
  - `data/added/standardized_datasets/工况21-24.csv`
- 清洗逻辑：复用 `src/current.data_loading`
- 窗口：`2s`
- 评估：
  - 对 `23` 个带标签工况做统一 case-level `LOOCV`
  - 分别汇报：
    - `final`
    - `added`
    - `all_labeled`

## 运行方式

```powershell
uv run python src/try/051_tcn_embedding_knn_residual/run_tcn_embedding_knn_residual.py
```

## 输出

- 输出目录：`outputs/try/051_tcn_embedding_knn_residual/`
- 主要文件：
  - `case_level_predictions.csv`
  - `summary_by_domain.csv`
  - `nearest_neighbors.csv`
  - `summary.md`
