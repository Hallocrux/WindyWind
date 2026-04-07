# 应变侧漂移缓解快速验证（2026-04-07）

- 状态：`current`
- 首次确认：`2026-04-07`
- 最近复核：`2026-04-07`
- 数据范围：
  - 主训练域：`data/final/` 的带标签工况
  - added 外部工况：`data/added/` 的 `工况21-24`
  - 重点工况：`工况22`
- 代码口径：
  - `src/try/038_strain_shift_mitigation_check/`
- 证据入口：
  - `outputs/try/038_strain_shift_mitigation_check/variant_config_table.csv`
  - `outputs/try/038_strain_shift_mitigation_check/case_level_predictions.csv`
  - `outputs/try/038_strain_shift_mitigation_check/summary.csv`
  - `outputs/try/038_strain_shift_mitigation_check/case22_focus.csv`
  - `outputs/try/038_strain_shift_mitigation_check/summary.md`

## 1. 目标

在 `037` 已确认 added 高估主要由应变侧输入驱动后，用最小实验矩阵验证：

- 只对应变做 `per-case zscore` 能否修复 added 外推；
- 只对应变做 `>2Hz high-pass` 能否修复 added 外推；
- 修复后的 `all_channels` 是否还能优于 `acc_only`。

## 2. 方法口径

- 评估集固定为：
  - `data/added/` 的 `工况21-24`
- 模型固定为：
  - `TinyTCN@5s`
- 训练池：
  - `full_final_pool`
  - `clean_final_pool`
- 变体矩阵：
  - `rpm_knn4`
  - `acc_only`
  - `all_channels_raw`
  - `all_channels + strain_case_zscore`
  - `all_channels + strain_highpass_2hz`
- 指标：
  - `case_mae`
  - `case_rmse`
  - `mean_signed_error`
  - `工况22 abs_error`

## 3. 当前结果

### 3.1 [2026-04-07] 两种应变修复都能大幅缓解原始全通道外推崩坏

- 状态：`current`
- 首次确认：`2026-04-07`
- 最近复核：`2026-04-07`

`full_final_pool` 下的关键对照：

- `all_channels_raw`
  - `case_mae = 3.1588`
  - `case22_abs_error = 4.7149`
  - `mean_signed_error = 3.1588`
- `all_channels + strain_case_zscore`
  - `case_mae = 0.4563`
  - `case22_abs_error = 1.1961`
  - `mean_signed_error = 0.4081`
- `all_channels + strain_highpass_2hz`
  - `case_mae = 0.4295`
  - `case22_abs_error = 0.8618`
  - `mean_signed_error = 0.2857`

这说明：

- added 外推里的应变问题不是完全不可修复；
- 无论做幅值标准化还是去低频，都会显著缓解系统性高估；
- 当前更有效的修复方向是：
  - 去掉应变侧 `0-2Hz` 低频成分
  - 而不是只做简单 `zscore`

### 3.2 [2026-04-07] `high-pass >2Hz` 是当前最有效的应变修复手段

- 状态：`current`
- 首次确认：`2026-04-07`
- 最近复核：`2026-04-07`

`full_final_pool` 下：

- `all_channels + strain_highpass_2hz`
  - `case_mae = 0.4295`
- `all_channels + strain_case_zscore`
  - `case_mae = 0.4563`

`工况22` 上：

- `highpass_2hz`
  - `pred = 4.2618`
  - `abs_error = 0.8618`
- `case_zscore`
  - `pred = 4.5961`
  - `abs_error = 1.1961`

这说明：

- 当前 added 应变失配的主要矛盾更像“低频机制偏移”；
- `工况22` 的异常确实能被“屏蔽低频”更有效地缓解。

### 3.3 [2026-04-07] 即使做了应变修复，当前 added 主线仍不应回到默认全通道

- 状态：`current`
- 首次确认：`2026-04-07`
- 最近复核：`2026-04-07`

最优结果对照：

- `rpm_knn4`
  - `case_mae = 0.2293`
- `full_final_pool | acc_only`
  - `case_mae = 0.3471`
  - `case22_abs_error = 0.2009`
- `full_final_pool | all_channels + strain_highpass_2hz`
  - `case_mae = 0.4295`
  - `case22_abs_error = 0.8618`

这说明：

- 应变高频修复后的 `all_channels` 已经从“不可用”回到“可参考”；
- 但它仍没有超过 `acc_only`；
- 因此当前 added 方向的默认外推主线，仍应优先采用：
  - `acc + rpm`
  - 而不是恢复到原始 `all_channels`

### 3.4 [2026-04-07] clean training pool 不比 full training pool 更优

- 状态：`current`
- 首次确认：`2026-04-07`
- 最近复核：`2026-04-07`

关键对照：

- `clean_final_pool | acc_only`
  - `case_mae = 0.4668`
- `clean_final_pool | all_channels + strain_highpass_2hz`
  - `case_mae = 0.4692`
- `full_final_pool | acc_only`
  - `case_mae = 0.3471`
- `full_final_pool | all_channels + strain_highpass_2hz`
  - `case_mae = 0.4295`

这说明：

- 在 added 外部验证上，继续砍掉难工况训练池并不能带来额外优势；
- 对当前这条线来说，训练池规模仍然重要。

## 4. 当前判断

`2026-04-07` 的这轮验证支持以下判断：

- 应变侧不是完全不可用，但必须先做额外处理；
- 当前最值得继续的应变修复方向是：
  - `>2Hz high-pass`
  - 或进一步细化为更聚焦的中频带策略；
- 在没有更强证据前，added 主线应优先切到：
  - `acc_only`
  - 或 `acc + rpm`
- 应变侧后续更适合作为“已知可部分修复的域适配分支”，而不是直接恢复到默认全通道主线。
