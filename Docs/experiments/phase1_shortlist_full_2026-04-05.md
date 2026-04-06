# 第一阶段 shortlist full 复核（2026-04-05）

- 状态：`historical`
- 首次确认：`2026-04-05`
- 最近复核：`2026-04-05`
- 数据范围：manifest 迁移后的 `工况1` 到 `工况20`
- 代码口径：
  - `src/try/010_phase1_shortlist_experiments/`
  - `src/try/009_phase1_feature_groups/`
- 证据入口：
  - `outputs/try/010_phase1_shortlist_experiments/full_best_model_summary.csv`
  - `outputs/try/010_phase1_shortlist_experiments/full_best_case_level_predictions.csv`
  - `outputs/try/010_phase1_shortlist_experiments/full_g1_model_summary.csv`
  - `outputs/model_summary.csv`

## 1. 目标

在停止第一阶段全矩阵实验之后，用固定 shortlist 组合快速判断：

- 哪个 `rpm-free` 特征组最值得继续保留；
- 是否已经出现明显优于当前正式 `Ridge + VIB_FT` 的候选。

本轮 shortlist 固定比较：

1. `Ridge + G3_CROSS_CHANNEL + rpm_free`
2. `Ridge + G1_ROBUST_TIME + rpm_free`
3. `Ridge + G6_TIME_FREQ_CROSS + rpm_free`
4. `Ridge + G3_CROSS_CHANNEL + rpm_aware`

## 2. 当前结论

### 2.1 [2026-04-05] shortlist 的 full 最优组合是 `Ridge + G6_TIME_FREQ_CROSS + rpm_free`

- 状态：`historical`
- 首次确认：`2026-04-05`
- 最近复核：`2026-04-05`

结论：

- `Ridge + G6_TIME_FREQ_CROSS + rpm_free` 的 full 结果为：
  - `case_mae = 0.4045`
  - `case_rmse = 0.7106`
  - `case_mape = 11.6894%`
- 当前正式基线 `Ridge + VIB_FT` 的结果为：
  - `case_mae = 0.4507`
  - `case_rmse = 0.8163`
  - `case_mape = 13.5355%`
- 因此本轮 shortlist 下，`G6_TIME_FREQ_CROSS` 已经明显优于当前正式 `rpm-free` 基线。

### 2.2 [2026-04-05] `G1_ROBUST_TIME` 的 full 提升有限，不足以替代 `G6`

- 状态：`historical`
- 首次确认：`2026-04-05`
- 最近复核：`2026-04-05`

结论：

- `Ridge + G1_ROBUST_TIME + rpm_free` 的 full 结果为：
  - `case_mae = 0.4442`
  - `case_rmse = 0.8331`
  - `case_mape = 13.4234%`
- 它只比当前正式 `Ridge + VIB_FT` 略好，但明显弱于 `Ridge + G6_TIME_FREQ_CROSS + rpm_free`。
- 因此 `G1_ROBUST_TIME` 更适合作为对照组，不适合作为当前 shortlist 冠军。

## 3. 按工况观察

### 3.1 `G6` 相比当前正式基线

- 改善工况数：`11`
- 变差工况数：`8`
- 恶化超过 `0.10 m/s` 的工况数：`2`
- 恶化超过 `0.20 m/s` 的工况数：`0`

改善更明显的工况包括：

- `工况1`
- `工况3`
- `工况5`
- `工况9`
- `工况15`

回退更明显的工况包括：

- `工况16`
- `工况17`
- `工况10`

### 3.2 `G6` 相比 `G1`

- `G6` 更好的工况数：`10`
- `G1` 更好的工况数：`9`

解释：

- `G6` 不是在所有工况上都胜出；
- 但它在若干关键工况上的改善幅度更大，因此整体 `case_mae / case_rmse` 都优于 `G1`。

## 4. 当前判断

`2026-04-05` 这轮 shortlist full 复核支持以下判断：

- 第一阶段如果继续做少量快速实验，默认特征组应优先保留 `G6_TIME_FREQ_CROSS`；
- 后续不值得再围绕 `G1` 单独投入大量比较成本；
- 下一步更合理的是固定 `G6_TIME_FREQ_CROSS + rpm_free`，再做少量模型 shortlist，而不是回到大矩阵。
