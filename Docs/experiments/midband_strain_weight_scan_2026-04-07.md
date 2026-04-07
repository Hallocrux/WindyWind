# 中频应变细扫与融合权重验证（2026-04-07）

- 状态：`current`
- 首次确认：`2026-04-07`
- 最近复核：`2026-04-07`
- 数据范围：
  - 主训练域：`data/final/` 的带标签工况
  - added 外部工况：`data/added/` 的 `工况21-24`
  - 重点工况：`工况22`
- 代码口径：
  - `src/try/040_midband_strain_weight_scan/`
- 证据入口：
  - `outputs/try/040_midband_strain_weight_scan/variant_config_table.csv`
  - `outputs/try/040_midband_strain_weight_scan/case_level_predictions.csv`
  - `outputs/try/040_midband_strain_weight_scan/summary.csv`
  - `outputs/try/040_midband_strain_weight_scan/case22_focus.csv`
  - `outputs/try/040_midband_strain_weight_scan/best_fusion_reference.csv`
  - `outputs/try/040_midband_strain_weight_scan/summary.md`

## 1. 目标

在 `039` 已经把 added 应变可迁移频带缩到 `3-6Hz` 邻域后，继续验证：

- `3-6Hz` 附近是否存在更稳的细频带；
- `acc + 中频应变` 的最佳融合权重落在哪里；
- 当前最佳方案是否已经超过 `rpm_knn4`。

## 2. 方法口径

- 训练池：
  - `full_final_pool`
- 模型：
  - `TinyTCN@5s`
- 细频带候选：
  - `2.5-5.5Hz`
  - `3.0-5.0Hz`
  - `3.0-6.0Hz`
  - `3.5-6.5Hz`
- 基线：
  - `rpm_knn4`
  - `acc_only`
- 融合：
  - `acc_only + strain(3.0-6.0Hz)`
  - 应变权重：`0.2 / 0.3 / 0.4 / 0.5`

## 3. 当前结果

### 3.1 [2026-04-07] 当前最优细频带已收敛到约 `3.0-6.0Hz`

- 状态：`current`
- 首次确认：`2026-04-07`
- 最近复核：`2026-04-07`

细频带结果：

- `strain_bandpass_3.0_6.0Hz`
  - `case_mae = 0.2584`
  - `case22_abs_error = 0.1324`
- `strain_bandpass_3.0_5.0Hz`
  - `case_mae = 0.2743`
- `strain_bandpass_3.5_6.5Hz`
  - `case_mae = 0.3670`
- `strain_bandpass_2.5_5.5Hz`
  - `case_mae = 0.4656`

这说明：

- `039` 的 `3-6Hz` 方向是对的；
- 更细的扫描后，当前 added 方向上最可迁移的应变频带收敛到约 `3.0-6.0Hz`；
- 过宽或偏移后的频带都会变差。

### 3.2 [2026-04-07] `3.0-6.0Hz` 单独使用时仍优于 `acc_only`

- 状态：`current`
- 首次确认：`2026-04-07`
- 最近复核：`2026-04-07`

关键对照：

- `full_final_pool | acc_only`
  - `case_mae = 0.4044`
  - `case22_abs_error = 0.5855`
- `full_final_pool | all_channels + strain_bandpass_3.0_6.0Hz`
  - `case_mae = 0.2584`
  - `case22_abs_error = 0.1324`

这说明：

- 对 added 外部集来说，保留 `3.0-6.0Hz` 中频应变依然是有价值的；
- 当前“默认只留 acc”已经不是最优的 learned 模型口径。

### 3.3 [2026-04-07] 当前最优融合权重落在较高应变占比，但整体仍未超过 `rpm_knn4`

- 状态：`current`
- 首次确认：`2026-04-07`
- 最近复核：`2026-04-07`

融合结果：

- `strain_weight = 0.5`
  - `case_mae = 0.2468`
  - `case22_abs_error = 0.3589`
- `strain_weight = 0.4`
  - `case_mae = 0.2696`
- `strain_weight = 0.3`
  - `case_mae = 0.2924`
- `strain_weight = 0.2`
  - `case_mae = 0.3152`

同时：

- `rpm_knn4`
  - `case_mae = 0.2293`

这说明：

- 在当前扫描内，融合权重呈现“应变权重越高越好”的趋势；
- 但即使是当前最优融合，也还没有超过 `rpm_knn4`；
- 因此 added 主线还不能下结论切到“加权 `acc + strain` 已经全面最优”。

### 3.4 [2026-04-07] `工况22` 仍然最支持保留 `3.0-6.0Hz` 中频应变

- 状态：`current`
- 首次确认：`2026-04-07`
- 最近复核：`2026-04-07`

`工况22` 的 case 级结果：

- `strain_bandpass_3.0_6.0Hz`
  - `pred = 3.5324`
  - `abs_error = 0.1324`
- `strain_bandpass_3.0_5.0Hz`
  - `pred = 3.6090`
  - `abs_error = 0.2090`
- `rpm_knn4`
  - `pred = 3.1342`
  - `abs_error = 0.2658`
- `fusion weight 0.5`
  - `pred = 3.7589`
  - `abs_error = 0.3589`
- `acc_only`
  - `pred = 3.9855`
  - `abs_error = 0.5855`

这说明：

- 如果只看 `工况22` 修复效果，单独 `3.0-6.0Hz` 中频应变仍然是当前最强方案；
- 融合更有利于整体 `21-24` 的平均表现，而不一定最利于 `工况22` 本身。

## 4. 当前判断

`2026-04-07` 的这轮细扫支持以下判断：

- added 方向上，当前最优细频带已经从粗粒度的 `3-6Hz` 收敛到更精确的 `3.0-6.0Hz`；
- `3.0-6.0Hz` 单独使用时，已经明显优于 `acc_only`；
- 当前最佳融合比重在这轮扫描中落在较高应变占比一侧，但整体仍未超过 `rpm_knn4`；
- 因此当前更稳妥的下一步应是：
  - 把 `rpm_knn4` 保留为 added 的强解析基线；
  - 把 `strain(3.0-6.0Hz)` 作为最优 learned 应变分支继续复核；
  - 如要继续融合，优先细化更高应变权重与更窄的 `3.0-6.0Hz` 邻域。
