# TinyTCN RPM 与风速窗长结果对照备注（2026-04-07）

- 状态：`current`
- 首次确认：`2026-04-07`
- 最近复核：`2026-04-07`
- 数据范围：manifest 迁移后的 `工况1` 到 `工况20`
- 代码口径：
  - 风速窗长扫描：`src/try/014_phase3_tcn_window_length_scan/`
  - RPM 细窗长扫描：`src/try/024_tinytcn_rpm_fine_window_scan/`
  - 风速多尺度快速验证：`src/try/026_tinytcn_priority1_quickcheck/`
- 证据入口：
  - `outputs/try/014_phase3_tcn_window_length_scan/tcn_window_scan_summary.csv`
  - `outputs/try/014_phase3_tcn_window_length_scan/tcn_window_scan_case_level_predictions.csv`
  - `outputs/try/024_tinytcn_rpm_fine_window_scan/rpm_fine_window_scan_summary.csv`
  - `outputs/try/024_tinytcn_rpm_fine_window_scan/rpm_fine_window_scan_case_level_predictions.csv`
  - `outputs/try/026_tinytcn_priority1_quickcheck/full19_multiscale_late_fusion_2s_8s_summary.csv`

## 1. 目标

对照 `2026-04-06` 已有的两类 TinyTCN 窗长结果：

- 风速回归
- RPM 回归

回答一个更具体的问题：

- RPM 的窗长偏好与风速不一致时，这条证据应当怎样用于当前风速多尺度主线。

## 2. 当前结果

### 2.1 [2026-04-07] RPM 与风速的单窗长最优点确实不同

- 状态：`current`
- 首次确认：`2026-04-07`
- 最近复核：`2026-04-07`

全量 summary 对照：

- 风速任务：
  - `TinyTCN@5s`
    - `case_mae = 0.3030`
  - `TinyTCN@4s`
    - `case_mae = 0.3115`
  - `TinyTCN@8s`
    - `case_mae = 0.3136`
  - `TinyTCN@2s`
    - `case_mae = 0.3236`
- RPM 任务：
  - `TinyTCN@3.0s`
    - `case_mae = 5.1863`
  - `TinyTCN@4.5s`
    - `case_mae = 7.7412`
  - `TinyTCN@2.0s`
    - `case_mae = 7.8917`
  - `TinyTCN@5.0s`
    - `case_mae = 8.3832`

这说明：

- RPM 任务更偏向 `2.5s - 4.5s` 这一段，最佳点落在 `3.0s`；
- 风速任务在单窗长下更偏向 `4s - 8s`，其中 `5s` 最优。

### 2.2 [2026-04-07] 按工况看，RPM 与风速也不是同一套窗长偏好

- 状态：`current`
- 首次确认：`2026-04-07`
- 最近复核：`2026-04-07`

按工况取各自最优窗长时，存在明显不一致，例如：

- `工况1`
  - RPM 最优：`3.0s`
  - 风速最优：`5s`
- `工况8`
  - RPM 最优：`2.0s`
  - 风速最优：`5s`
- `工况14`
  - RPM 最优：`3.0s`
  - 风速最优：`8s`
- `工况18`
  - RPM 最优：`4.0s`
  - 风速最优：`8s`

也存在局部一致的工况，例如：

- `工况3`
  - RPM 最优：`2.0s`
  - 风速最优：`2s`
- `工况16`
  - RPM 最优：`3.0s`
  - 风速最优：`4s`
- `工况20`
  - RPM 最优：`2.5s`
  - 风速最优：`2s`

这说明：

- RPM 窗长结论不能直接平移成风速任务的默认窗长；
- 但它提供了一个稳定信号：较短或中等窗长里确实含有一类对转速更敏感的信息。

### 2.3 [2026-04-07] RPM 窗长实验更支持“多尺度互补”，而不是“风速应直接改用 3s”

- 状态：`current`
- 首次确认：`2026-04-07`
- 最近复核：`2026-04-07`

`2026-04-06` 的风速多尺度快速验证已经显示：

- `TinyTCN@5s`
  - `case_mae = 0.3030`
- `TinyTCN_multiscale_late_fusion_2s_8s`
  - `case_mae = 0.1685`

把这个结果与 RPM 窗长实验合并理解，更合理的解释是：

- 风速任务并不只依赖一种时间尺度；
- 较短或中等窗长更可能携带与旋转周期、局部节律相关的信息；
- 较长窗长更可能提供更稳的慢变量、工况级能量结构或更长时间上下文；
- 因此当前更应把 RPM 窗长实验视为“短/中窗分支有物理依据”，而不是“风速主窗长应直接从 `5s` 改成 `3s`”。

## 3. 当前判断

`2026-04-07` 的这条对照备注支持以下判断：

- RPM 结果与风速结果不同，这不是冲突，而是多尺度路线的重要旁证；
- 风速主线的下一步不应简单把单窗长改成 `3s`；
- 更合理的后续候选是：
  - 保留一个较长窗分支；
  - 再补一个受 RPM 结果启发的短/中窗分支；
  - 优先比较 `2s+8s`、`3s+8s` 或 `3s+5s` 这类联合结构。
