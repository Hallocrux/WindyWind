# 009 第一阶段特征组筛选

## 目标

- 基于当前 `src/current/` 表格主线的清洗与切窗口径，完成第一阶段的特征组筛选、finalist 确认和晋升判定。
- 本探索固定采用“两段式”特征提取：
  - baseline block：从原始窗口提 `raw_mean` 与 `raw_median`
  - dynamic block：先对每通道减去该窗口 `raw_mean`，再提动态统计、频域和跨通道特征
- 第一轮只比较 `rpm-free` 的 `Ridge` 结果；`rpm-aware` 只对 finalist 补跑。

## 输入与口径

- 数据入口：
  - `data/final/dataset_manifest.csv`
  - `data/final/datasets/工况*.csv`
- 清洗逻辑：复用 `src/current.data_loading`
- 默认窗口参数：
  - 采样率 `50Hz`
  - 窗长 `5s`（250 点）
  - 步长 `2.5s`（125 点）
- 评估方式：
  - 分组单位为工况
  - `Leave-One-Condition-Out`

## 运行方式

```powershell
uv run python src/try/009_phase1_feature_groups/run_phase1.py
```

如需显式指定输出目录：

```powershell
uv run python src/try/009_phase1_feature_groups/run_phase1.py --output-dir outputs/try/009_phase1_feature_groups
```

开发态推荐先跑固定小数据集：

```powershell
uv run python src/try/009_phase1_feature_groups/run_phase1.py --mode dev --max-workers 4 --rf-n-jobs 1
```

如需显式指定工况：

```powershell
uv run python src/try/009_phase1_feature_groups/run_phase1.py --case-ids 1 2 3 5 15 16 --max-workers 4
```

## 输出

- 输出目录：`outputs/try/009_phase1_feature_groups/`
- 固定产物：
  - `feature_group_summary.csv`
  - `case_level_screening.csv`
  - `finalist_model_summary.csv`
  - `finalist_case_level_predictions.csv`
  - `feature_manifest.csv`
  - `phase1_case_mae_bar.png`
  - `phase1_case_delta_heatmap.png`
  - `summary.md`

## 判定规则

- 第一轮对 `G0` 到 `G7` 只跑 `Ridge + rpm-free`
- finalist 资格固定为：
  - 相比 `G0_BASE`，`case_mae` 至少下降 `0.01`
  - 相比 `G0_BASE`，`case_rmse` 不上升
- 只对第一名 finalist 做晋升判定
- 若未通过严格晋升门槛，则只保留探索结论，不更新 `src/current/`
