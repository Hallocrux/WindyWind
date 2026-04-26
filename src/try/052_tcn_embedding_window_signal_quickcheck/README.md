# 052 TinyTCN embedding window signal quickcheck

## 目标

- 不做全量 `LOOCV`，只对代表性 holdout 工况做小规模信号验证；
- 比较 `2s`、`5s` 与 `2s+8s` 这三类 embedding residual 是否比 `051` 的 `2s` 更稳；
- 回答一个更具体的问题：
  - 对 `rpm_knn4` 主干来说，哪类时序窗长更适合作为局部修正信号。

## holdout 工况

- `final` 难工况：
  - `工况1`
  - `工况3`
  - `工况17`
  - `工况18`
- `added`：
  - `工况21`
  - `工况22`
  - `工况23`
  - `工况24`

## 输入与口径

- 主训练域：
  - `data/final/dataset_manifest.csv`
  - `data/final/datasets/工况*.csv`
- 补充域：
  - `data/added/dataset_manifest.csv`
  - `data/added/standardized_datasets/工况21-24.csv`
- 清洗逻辑：复用 `src/current.data_loading`
- 窗口：
  - `2s`
  - `5s`
  - `8s`
- 评估：
  - 对指定 `8` 个 holdout 工况分别做 leave-one-case-out 预测
  - 汇报：
    - `final_focus`
    - `added_focus`
    - `focus_all`

## 运行方式

```powershell
uv run python src/try/052_tcn_embedding_window_signal_quickcheck/run_tcn_embedding_window_signal_quickcheck.py
```

## 输出

- 输出目录：`outputs/try/052_tcn_embedding_window_signal_quickcheck/`
- 主要文件：
  - `case_level_predictions.csv`
  - `summary_by_domain.csv`
  - `nearest_neighbors.csv`
  - `summary.md`
