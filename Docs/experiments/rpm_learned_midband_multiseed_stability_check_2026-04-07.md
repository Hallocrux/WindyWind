# `rpm_knn4 + TinyTCN midband` 多随机种子稳定性复核（2026-04-07）

- 状态：`current`
- 首次确认：`2026-04-07`
- 最近复核：`2026-04-07`
- 数据范围：
  - 主训练域：`data/final/` 的带标签工况
  - added 外部工况：`data/added/` 的 `工况21-24`
  - 重点工况：`工况22`
- 代码口径：
  - `src/try/042_rpm_learned_midband_multiseed_stability_check/`
- 证据入口：
  - `outputs/try/042_rpm_learned_midband_multiseed_stability_check/variant_config_table.csv`
  - `outputs/try/042_rpm_learned_midband_multiseed_stability_check/seed_summary.csv`
  - `outputs/try/042_rpm_learned_midband_multiseed_stability_check/seed_case_level_predictions.csv`
  - `outputs/try/042_rpm_learned_midband_multiseed_stability_check/best_variant_by_seed.csv`
  - `outputs/try/042_rpm_learned_midband_multiseed_stability_check/stability_overview.csv`
  - `outputs/try/042_rpm_learned_midband_multiseed_stability_check/pairwise_comparison.csv`
  - `outputs/try/042_rpm_learned_midband_multiseed_stability_check/case22_by_seed.csv`
  - `outputs/try/042_rpm_learned_midband_multiseed_stability_check/summary.md`

## 1. 目标

对 `041` 中单次最优的 `rpm_knn4 + TinyTCN all_channels midband @ learned_weight=0.5` 做多随机种子复核，确认两个问题：

- 混合方案是否真的稳定优于 `rpm_knn4`；
- `w=0.5` 是否真的是稳定最优，而不是单次 seed 的偶然点。

## 2. 方法口径

- 训练池：
  - `full_final_pool`
- learned 分支：
  - `TinyTCN@5s`
  - `all_channels + strain bandpass 3.0-6.0Hz`
- 对照变体：
  - `rpm_knn4`
  - `TinyTCN all_channels midband`
  - `rpm_knn4 + TinyTCN all_channels midband @ w=0.3`
  - `rpm_knn4 + TinyTCN all_channels midband @ w=0.5`
  - `rpm_knn4 + TinyTCN all_channels midband @ w=0.7`
- base seeds：
  - `42, 52, 62, 72, 82, 92, 102, 112, 122, 132`

## 3. 当前结果

### 3.1 [2026-04-07] 混合路线本身已经通过多 seed 复核，不是单次偶然收益

- 状态：`current`
- 首次确认：`2026-04-07`
- 最近复核：`2026-04-07`

跨 seed 聚合结果：

- `rpm_knn4`
  - `case_mae mean = 0.2293`
- `TinyTCN all_channels midband`
  - `case_mae mean = 0.3175`
  - `better_than_rpm_rate = 0.1`
- `fusion @ w=0.3`
  - `case_mae mean = 0.1627`
  - `case_mae std = 0.0223`
  - `better_than_rpm_rate = 1.0`
- `fusion @ w=0.5`
  - `case_mae mean = 0.1822`
  - `case_mae std = 0.0341`
  - `better_than_rpm_rate = 0.9`

这说明：

- `rpm + learned midband` 的提升不是单次 seed 才成立；
- 固定权重在 `0.3` 或 `0.5` 时，整体都明显优于 `rpm_knn4`；
- 纯 learned 分支本身并不稳，真正稳的是“解析 + learned”组合。

### 3.2 [2026-04-07] `w=0.5` 没有通过“稳定最优”复核，更稳的固定权重落在 `0.3`

- 状态：`current`
- 首次确认：`2026-04-07`
- 最近复核：`2026-04-07`

跨 seed 固定权重对照：

- `fusion @ w=0.3`
  - `case_mae mean = 0.1627`
  - `best_seed_count = 7 / 10`
- `fusion @ w=0.5`
  - `case_mae mean = 0.1822`
  - `best_seed_count = 3 / 10`
- `fusion @ w=0.7`
  - `case_mae mean = 0.2309`
  - `best_seed_count = 0 / 10`

相对对照：

- `w=0.5` 对 `w=0.3`
  - `reference_better_case_mae_count = 3 / 10`
  - `delta_case_mae_mean = +0.0195`
- `w=0.5` 对 `w=0.7`
  - `reference_better_case_mae_count = 10 / 10`
  - `delta_case_mae_mean = -0.0486`

这说明：

- `041` 中的 `w=0.5` 更像是单次 seed 的局部最优；
- 如果目标是 `工况21-24` 的整体 `case_mae` 稳定性，当前固定权重应下修到更保守的 `0.3` 邻域；
- `w=0.7` 已可视作过重依赖 learned 分支，不宜作为默认候选。

### 3.3 [2026-04-07] `w=0.5` 虽然不是整体最稳，但对 `工况22` 的修复仍更强

- 状态：`current`
- 首次确认：`2026-04-07`
- 最近复核：`2026-04-07`

`工况22` 聚合结果：

- `rpm_knn4`
  - `case22_abs_error mean = 0.2658`
- `fusion @ w=0.3`
  - `case22_abs_error mean = 0.1431`
  - `better_than_rpm_case22_rate = 1.0`
- `fusion @ w=0.5`
  - `case22_abs_error mean = 0.0885`
  - `better_than_rpm_case22_rate = 1.0`

这说明：

- 如果只看 `工况22` 这类 hardest case，`w=0.5` 仍然比 `w=0.3` 更有修复力；
- 当前最合理的解释是：
  - `w=0.3` 更偏整体稳健；
  - `w=0.5` 更偏对 hardest case 的激进修复。

## 4. 当前判断

`2026-04-07` 的多随机种子复核支持以下当前结论：

- added 方向上，“`rpm_knn4 + learned midband` 混合”已经被确认不是偶然收益，而是稳定成立的路线；
- 但 `w=0.5` 没有被确认成“真正稳定最优”的固定权重；
- 如果目标是 added 外部集 `工况21-24` 的整体平均误差，当前更稳的默认权重应放在 `0.3` 邻域；
- 如果目标是尽量修平 `工况22` 这类 hardest case，`w=0.5` 仍可保留为参考权重；
- 因此这条线的当前默认候选应更新为：
  - `rpm_knn4 + TinyTCN all_channels midband`
  - 默认固定权重优先参考 `0.3`
  - `0.5` 作为 case22-oriented 备选参考
