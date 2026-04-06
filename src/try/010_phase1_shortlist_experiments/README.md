# 010 第一阶段 shortlist 实验

## 目标

- 停止大矩阵实验，只运行少量最有可能提升 `rpm-free` 主线的组合。
- 复用 `009_phase1_feature_groups` 已实现的“两段式”特征提取逻辑：
  - baseline block：原始窗口的 `raw_mean`、`raw_median`
  - dynamic block：每通道先减去窗口 `raw_mean`，再提动态特征
- 当前固定只跑 4 个 shortlist 组合，不再扩展到全矩阵。

## 固定实验列表

1. `Ridge + G3_CROSS_CHANNEL + rpm_free`
2. `Ridge + G1_ROBUST_TIME + rpm_free`
3. `Ridge + G6_TIME_FREQ_CROSS + rpm_free`
4. `Ridge + G3_CROSS_CHANNEL + rpm_aware`

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

开发态先跑固定小数据集：

```powershell
uv run python src/try/010_phase1_shortlist_experiments/run_shortlist.py --mode dev
```

全量运行：

```powershell
uv run python src/try/010_phase1_shortlist_experiments/run_shortlist.py --mode full
```

显式指定工况：

```powershell
uv run python src/try/010_phase1_shortlist_experiments/run_shortlist.py --case-ids 1 2 3 5 15 16
```

## 输出

- 输出目录：`outputs/try/010_phase1_shortlist_experiments/`
- 固定产物：
  - `shortlist_model_summary.csv`
  - `shortlist_case_level_predictions.csv`
  - `shortlist_feature_manifest.csv`
  - `summary.md`
