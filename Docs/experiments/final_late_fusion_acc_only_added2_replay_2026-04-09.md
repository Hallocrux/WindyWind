# final 训练的 acc-only 2s+8s 晚融合外推 added2（2026-04-09）

- 状态：`current`
- 首次确认：`2026-04-09`
- 最近复核：`2026-04-09`
- 数据范围：
  - 训练域：`final` 带标签工况 `1, 3-20`
  - 外部域：`data/added/` 的 `工况21-24`
  - 外部域：`data/added2/` 的 `工况25-30`
- 代码口径：
  - `src/try/081_final_late_fusion_acc_only_added2_replay/`
- 证据入口：
  - `outputs/try/081_final_late_fusion_acc_only_added2_replay/case_level_predictions.csv`
  - `outputs/try/081_final_late_fusion_acc_only_added2_replay/summary_by_domain.csv`
  - `outputs/try/081_final_late_fusion_acc_only_added2_replay/summary_by_domain_with_063_reference.csv`
  - `outputs/try/081_final_late_fusion_acc_only_added2_replay/case_compare_with_063_reference.csv`
  - `outputs/try/081_final_late_fusion_acc_only_added2_replay/summary.md`

## 1. 目标

补测此前未单独验证的一条 direct learned 路线：

- `final` 训练的 `TinyTCN@2s`
- `final` 训练的 `TinyTCN@8s`
- `final` 训练的 `TinyTCN 2s+8s late fusion`

但输入只保留：

- `acc_only`

本轮问题是：

- 把 `all_channels` 改成 `acc_only` 后，`2s+8s` 晚融合是否能在 `added / added2` 上恢复可用；
- 它是否能改写 `063` 中“direct 2s+8s 无法外推”的结论。

## 2. 方法口径

- 训练池：
  - `final` 带标签工况 `1, 3-20`
- 外部回放：
  - `added(21-24)`
  - `added2(25-30)`
- 输入：
  - 只保留加速度通道 `WSMS*.Acc*`
- 窗口：
  - `2s`
  - `8s`
  - 工况级 `2s+8s` 晚融合
- 对照：
  - `063` 中同训练池、同评估域的 `all_channels` 版本

## 3. 当前结果

### 3.1 [2026-04-09] `acc_only` 明显优于 `all_channels`

`added2` 对照：

- `081 | acc_only_2s`
  - `case_mae = 1.7982`
- `081 | acc_only_8s`
  - `case_mae = 1.7812`
- `081 | acc_only_2s_8s_fusion`
  - `case_mae = 1.7897`
- `063 | all_channels_2s`
  - `case_mae = 2.1985`
- `063 | all_channels_8s`
  - `case_mae = 2.4157`
- `063 | all_channels_2s_8s_fusion`
  - `case_mae = 2.2897`

`added` 对照：

- `081 | acc_only_8s`
  - `case_mae = 0.2201`
- `081 | acc_only_2s_8s_fusion`
  - `case_mae = 0.2279`
- `063 | all_channels_2s_8s_fusion`
  - `case_mae = 3.5497`

这说明：

- `063` 的 direct learned 外推崩坏，确实很大一部分来自应变侧；
- 把输入切到 `acc_only` 后，`added / added2` 都明显更稳；
- 因而“all_channels 的失败”不能简单等价成“direct TinyTCN 完全没救”。

### 3.2 [2026-04-09] 但 `acc_only 2s+8s` 仍然没有成为 `added2` 可用主线

`added2` 上，本轮最优是：

- `acc_only_8s`
  - `case_mae = 1.7812`

而不是：

- `acc_only_2s_8s_fusion`
  - `case_mae = 1.7897`

并且对照当前更稳的解析主干：

- `rpm_knn4`
  - `case_mae = 0.8131`

这说明：

- `acc_only` 虽然修掉了大部分 `all_channels` 崩坏；
- 但 `2s+8s` 晚融合并没有因此成为 `added2` 的默认外推解；
- 在当前口径下，direct learned 仍明显弱于 `rpm-first`。

### 3.3 [2026-04-09] `2s+8s` 在 acc-only 下也没有显示出比单独 `8s` 更强的外部域价值

`added2`：

- `acc_only_8s`
  - `case_mae = 1.7812`
- `acc_only_2s_8s_fusion`
  - `case_mae = 1.7897`
- `acc_only_2s`
  - `case_mae = 1.7982`

`added`：

- `acc_only_8s`
  - `case_mae = 0.2201`
- `acc_only_2s_8s_fusion`
  - `case_mae = 0.2279`
- `acc_only_2s`
  - `case_mae = 0.4000`

这说明：

- 在外部域上，`8s` 单窗长比 `2s+8s` 晚融合略稳；
- `2s+8s` 在 `final` 域里成立的互补性，并没有自然迁移成 `acc_only` 外部域优势；
- 当前更像“长窗更稳”，而不是“多尺度更稳”。

### 3.4 [2026-04-09] `added2` 的最难工况仍然没有被 direct acc-only 路线解决

`acc_only_2s_8s_fusion` 的 `added2` case 级误差：

- `工况25`
  - `abs_error = 2.8604`
- `工况26`
  - `abs_error = 2.0039`
- `工况27`
  - `abs_error = 0.9158`
- `工况28`
  - `abs_error = 1.7823`
- `工况29`
  - `abs_error = 2.0253`
- `工况30`
  - `abs_error = 1.1504`

这说明：

- `acc_only` 只是把 direct learned 从“系统性大崩坏”拉回到“仍然偏弱”；
- 它没有解决 `added2` 高风速与低风速边界 case 的核心泛化问题；
- 因此不能把它升级成新的 external 默认主线。

## 4. 当前判断

截至 `2026-04-09`，本轮更合理的结论应表达为：

- `2s+8s late fusion + acc_only` 确实测过了；
- 它明显优于 `all_channels`；
- 但它仍明显弱于 `rpm_knn4`；
- 并且在 `acc_only` 下，`2s+8s` 也没有优于单独 `8s`。

## 5. 一句话版结论

截至 `2026-04-09`，`final` 训练的 `TinyTCN 2s+8s` 晚融合如果只保留 `acc_only`，在 `added / added2` 上会比 `all_channels` 稳得多；但它在 `added2` 上仍然不够好，且不如单独 `8s`，更远不如 `rpm-first` 主干，因此这条线只能保留为“direct learned 可参考修正版”，不能升级为默认外推主线。
