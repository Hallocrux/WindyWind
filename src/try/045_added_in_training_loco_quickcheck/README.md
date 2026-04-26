# 045 added 并入训练池后的 2s/8s LOCO quickcheck

## 目标

- 做一轮最小成本的信号验证；
- 把 `final` 带标签工况与 `added 21-24` 合成统一带标签池；
- 只复用现有 `TinyTCN@2s`、`TinyTCN@8s` 与工况级 `2s+8s` 晚融合；
- 观察 added 并入训练后，统一 `LOCO` 下的：
  - `final` 子集表现
  - `added` 子集表现
  - `all_labeled` 整体表现

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
  - `8s`
- 融合：
  - `pred_fusion = 0.5 * pred_2s + 0.5 * pred_8s`

## 运行方式

```powershell
.venv\Scripts\python.exe src/try/045_added_in_training_loco_quickcheck/run_added_in_training_loco_quickcheck.py
```

## 输出

- 输出目录：`outputs/try/045_added_in_training_loco_quickcheck/`
- 固定产物：
  - `case_level_predictions.csv`
  - `summary_by_domain.csv`
  - `summary.md`
