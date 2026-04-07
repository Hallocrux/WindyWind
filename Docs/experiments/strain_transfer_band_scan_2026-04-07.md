# 应变可迁移频带筛选（2026-04-07）

- 状态：`current`
- 首次确认：`2026-04-07`
- 最近复核：`2026-04-07`
- 数据范围：
  - 主训练域：`data/final/` 的带标签工况
  - added 外部工况：`data/added/` 的 `工况21-24`
  - 重点工况：`工况22`
- 代码口径：
  - `src/try/039_strain_transfer_band_scan/`
- 证据入口：
  - `outputs/try/039_strain_transfer_band_scan/variant_config_table.csv`
  - `outputs/try/039_strain_transfer_band_scan/case_level_predictions.csv`
  - `outputs/try/039_strain_transfer_band_scan/summary.csv`
  - `outputs/try/039_strain_transfer_band_scan/case22_focus.csv`
  - `outputs/try/039_strain_transfer_band_scan/best_band_reference.csv`
  - `outputs/try/039_strain_transfer_band_scan/summary.md`

## 1. 目标

在 `038` 已证明应变高通可以部分修复 added 外推后，继续验证：

- 是否存在比简单高通更可迁移的应变中频带；
- 该频带能否超过 `acc_only`；
- 它更适合作为单独输入，还是与 `acc_only` 做晚融合。

## 2. 方法口径

- 评估集：
  - `data/added/` 的 `工况21-24`
- 训练池：
  - `full_final_pool`
- 模型：
  - `TinyTCN@5s`
- 变体矩阵：
  - `rpm_knn4`
  - `acc_only`
  - `all_channels + strain_highpass_2hz`
  - `all_channels + strain_bandpass_2_4hz`
  - `all_channels + strain_bandpass_2_5hz`
  - `all_channels + strain_bandpass_3_6hz`
  - `all_channels + strain_bandpass_4_8hz`
  - `acc_only + strain_highpass_2hz` 晚融合
  - `acc_only + best_bandpass` 晚融合
- 指标：
  - `case_mae`
  - `case_rmse`
  - `mean_signed_error`
  - `工况22 abs_error`

## 3. 当前结果

### 3.1 [2026-04-07] 当前最可迁移的应变频带落在 `3-6Hz`

- 状态：`current`
- 首次确认：`2026-04-07`
- 最近复核：`2026-04-07`

频带扫描结果：

- `strain_bandpass_3_6hz`
  - `case_mae = 0.2880`
  - `case22_abs_error = 0.0024`
- `strain_highpass_2hz`
  - `case_mae = 0.3439`
  - `case22_abs_error = 0.6349`
- `strain_bandpass_4_8hz`
  - `case_mae = 0.3693`
- `strain_bandpass_2_4hz`
  - `case_mae = 0.4322`
- `strain_bandpass_2_5hz`
  - `case_mae = 0.4577`

这说明：

- 当前 added 方向上，最稳定的应变可迁移信息并不在“全部高频”，而更集中在约 `3-6Hz`；
- `工况22` 的异常几乎可以被 `3-6Hz` 频带单独修平；
- 因此后续应变侧探索应优先围绕中频带展开，而不是继续使用原始全频输入。

### 3.2 [2026-04-07] `3-6Hz` 单独使用时已经优于 `acc_only`

- 状态：`current`
- 首次确认：`2026-04-07`
- 最近复核：`2026-04-07`

关键对照：

- `full_final_pool | acc_only`
  - `case_mae = 0.3315`
  - `case22_abs_error = 0.6588`
- `full_final_pool | all_channels + strain_bandpass_3_6hz`
  - `case_mae = 0.2880`
  - `case22_abs_error = 0.0024`

这说明：

- 到 `2026-04-07` 为止，added 方向上已经不能再简单说“默认只保留 acc”；
- 更准确的表述应是：
  - 原始应变不能直接用
  - 但经过 `3-6Hz` 带通后，应变已经具备独立增益

### 3.3 [2026-04-07] `acc_only + 3-6Hz band-pass` 晚融合当前整体最优

- 状态：`current`
- 首次确认：`2026-04-07`
- 最近复核：`2026-04-07`

最优结果对照：

- `fusion_acc_only__full_final_pool|all_channels_strain_bandpass_3_6hz`
  - `case_mae = 0.2073`
  - `case_rmse = 0.2307`
  - `case22_abs_error = 0.3282`
- `rpm_knn4`
  - `case_mae = 0.2293`
- `strain_bandpass_3_6hz`
  - `case_mae = 0.2880`
- `acc_only`
  - `case_mae = 0.3315`

这说明：

- 当前最优 added 外推已经不再是单独 `rpm_knn4`；
- 也不是单独 `acc_only`；
- 而是：
  - `acc_only + strain(3-6Hz)` 晚融合

### 3.4 [2026-04-07] `3-6Hz` 频带在 added 上同时兼顾整体误差与 `工况22` 修复

- 状态：`current`
- 首次确认：`2026-04-07`
- 最近复核：`2026-04-07`

`工况22` 的 case 级结果：

- `strain_bandpass_3_6hz`
  - `pred = 3.3976`
  - `abs_error = 0.0024`
- `rpm_knn4`
  - `pred = 3.1342`
  - `abs_error = 0.2658`
- `fusion_acc_only__strain_bandpass_3_6hz`
  - `pred = 3.7282`
  - `abs_error = 0.3282`
- `acc_only`
  - `pred = 4.0588`
  - `abs_error = 0.6588`

这说明：

- 如果目标是最大限度修复 `工况22`，单独 `3-6Hz` 频带是当前最强方案；
- 如果目标是兼顾 `21-24` 整体稳定性，`acc + 3-6Hz` 晚融合更优；
- 这两条结果并不矛盾，反而说明：
  - `3-6Hz` 确实承载了可迁移的 added 应变信息

## 4. 当前判断

`2026-04-07` 的这轮扫描支持以下判断：

- added 方向上，当前最值得保留的应变信息集中在约 `3-6Hz`；
- 原始应变仍不应直接回到默认全通道主线；
- 但“应变整体先下线”的结论也应更新为：
  - 全频应变先下线
  - `3-6Hz` 中频应变保留为有效辅助分支；
- 如果继续推进，下一步最合理的方向是：
  - 以 `3-6Hz` 为中心做更细频带扫描
  - 或直接把 `acc + strain(3-6Hz)` 作为 added 主线候选继续复核。
