# 011 第一阶段 shortlist 树模型比较

## 目标

- 复用 `Try 10` 的 shortlist 特征组合，但把线性模型换成树模型做快速比较。
- 停止“特征组 × 模型 × task_mode”大矩阵，只保留：
  - 固定 shortlist 组合：
    1. `G3_CROSS_CHANNEL + rpm_free`
    2. `G1_ROBUST_TIME + rpm_free`
    3. `G6_TIME_FREQ_CROSS + rpm_free`
    4. `G3_CROSS_CHANNEL + rpm_aware`
  - 固定树模型：
    - `RandomForestRegressor`
    - `HistGradientBoostingRegressor`

## 输入与口径

- 数据入口：
  - `data/final/dataset_manifest.csv`
  - `data/final/datasets/工况*.csv`
- 清洗逻辑：复用 `src/current.data_loading`
- 特征逻辑：复用 `src/try/009_phase1_feature_groups/phase1_feature_groups_lib.py`
- 默认窗口参数：
  - 采样率 `50Hz`
  - 窗长 `5s`
  - 步长 `2.5s`
- 评估方式：
  - 分组单位为工况
  - `Leave-One-Condition-Out`

## 运行方式

开发态：

```powershell
uv run python src/try/011_phase1_shortlist_models/run_shortlist_models.py --mode dev --max-workers 2 --rf-n-jobs 1
```

全量：

```powershell
uv run python src/try/011_phase1_shortlist_models/run_shortlist_models.py --mode full --max-workers 2 --rf-n-jobs 1
```

显式指定工况：

```powershell
uv run python src/try/011_phase1_shortlist_models/run_shortlist_models.py --case-ids 1 2 3 5 15 16 --max-workers 2
```

## 输出

- 输出目录：`outputs/try/011_phase1_shortlist_models/`
- 固定产物：
  - `shortlist_tree_models_summary.csv`
  - `shortlist_tree_models_case_level_predictions.csv`
  - `summary.md`
