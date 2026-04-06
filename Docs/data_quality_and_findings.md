# 数据质量与稳定结论

本文档只记录已经形成证据链的事实与结论。  
每条结论都显式标注时间、状态、数据范围和代码口径。

## 1. [2026-04-05] 当前 20 工况主数据已切换为 manifest + 标准文件名

- 状态：`current`
- 首次确认：`2026-04-05`
- 最近复核：`2026-04-05`
- 数据范围：`工况1` 到 `工况20`
- 代码口径：`src/current.data_loading`
- 证据入口：
  - `data/final/dataset_manifest.csv`
  - `outputs/dataset_inventory.csv`

结论：

- 当前主数据已经不再从文件名解析工况标签
- `data/final/dataset_manifest.csv` 是唯一人工元数据来源
- 标准数据文件名统一为 `工况{ID}.csv`

## 2. [2026-04-05] 宽表主采样率仍稳定在约 50Hz

- 状态：`current`
- 首次确认：`2026-04-02`
- 最近复核：`2026-04-05`
- 数据范围：`工况1` 到 `工况20`
- 代码口径：`src/current.data_loading`
- 证据入口：
  - `outputs/dataset_inventory.csv`
  - `outputs/data_quality_summary.csv`

结论：

- 各工况的 `sampling_hz_est` 当前稳定在约 `50Hz`
- 当前主时间间隔仍可按 `0.02s` 处理

## 3. [2026-04-05] `WSMS00005.*` 仍应视为错误数据

- 状态：`current`
- 首次确认：`2026-04-02`
- 最近复核：`2026-04-05`
- 数据范围：至少覆盖 `工况5`、`工况12`、`工况13`、`工况14`
- 代码口径：`src/current.data_loading`
- 证据入口：
  - `outputs/dataset_inventory.csv`
  - `PROJECT.md` 历史结论迁移记录

结论：

- `WSMS00005.AccX/Y/Z` 当前继续视为错误列
- 正确的第 5 个加速度传感器信号实际在 `WSMS00006.*`
- 当前正式清洗必须忽略 `WSMS00005.*`

## 4. [2026-04-05] 新增工况 15-17 的缺失水平显著高于旧工况

- 状态：`current`
- 首次确认：`2026-04-05`
- 最近复核：`2026-04-05`
- 数据范围：`工况1` 到 `工况20`
- 代码口径：`src/current.data_quality`
- 证据入口：
  - `outputs/data_quality_summary.csv`

结论：

- 当前 20 个工况在共有有效信号通道上的平均缺失率为 `2.0493%`
- 平均首尾连续缺失删除比例为 `4.0543%`
- 缺失率最高的是 `工况15`，为 `6.1709%`
- 最长连续缺失段来自 `工况16`，长度 `352` 点，约 `7.04s`
- 当前新清洗口径下，保留窗口中的受缺失影响窗口占比为 `0.0000%`
- `工况15`、`工况16`、`工况17` 的缺失水平显著高于旧工况，后续值得单独做阶段性质量诊断

## 5. [2026-04-05] 当前默认表格主线的最佳模型已切换为 Ridge + VIB_FT_RPM

- 状态：`current`
- 首次确认：`2026-04-05`
- 最近复核：`2026-04-05`
- 数据范围：manifest 迁移后的 `工况1` 到 `工况20`
- 代码口径：`src/current/`
- 证据入口：
  - `outputs/model_summary.csv`
  - `outputs/unlabeled_predictions.csv`

结论：

- 当前带标签工况共有 `19` 个
- 当前总窗口数为 `866`
- 全局最优模型为 `Ridge + VIB_FT_RPM`
- 最佳 `rpm-free` 模型为 `Ridge + VIB_FT`
- `工况2.csv` 的当前无标签预测结果为 `3.3279 m/s`

## 6. [2026-04-04] 删除首尾连续缺失段后，边界段仍常与中段分离

- 状态：`historical`
- 首次确认：`2026-04-04`
- 最近复核：`2026-04-04`
- 数据范围：当时的 `14` 个工况
- 代码口径：
  - `src/current.data_loading`
  - `src/try/004_trimmed_boundary_stability_check/`
- 证据入口：
  - `outputs/try/004_trimmed_boundary_stability_check/`

结论：

- 删除首尾连续缺失段后，边界段的脏窗口问题显著缓解
- 但边界段与中段的特征分布差异并未完全消失
- 因此边界段不应直接等价视为“稳态中段”

备注：

- 这条结论仍可作为当前分析的背景事实
- 但它基于旧的 `14` 工况数据范围，不应直接替代对 `20` 工况口径的重新复核

## 7. [2026-04-04] 开始段和结束段相对中段普遍偏离

- 状态：`historical`
- 首次确认：`2026-04-04`
- 最近复核：`2026-04-04`
- 数据范围：当时的 `14` 个工况
- 代码口径：`src/try/003_start_end_segment_diagnosis/`
- 证据入口：
  - `outputs/try/003_start_end_segment_diagnosis/`

结论：

- 在旧 14 工况口径下，开始段和结束段相对中段普遍存在显著分布偏离
- 这说明边界段可能代表另一种阶段，而不是简单的噪声污染

替代关系：

- 该结论尚未被 20 工况口径下的同类复核替代
- 后续若有新的全量复核，应补一条新的 `current` 结论，并将本条改为 `superseded`

## 8. [2026-04-05] 去转频后，应变侧候选结构频率在多数工况上集中于约 2.2Hz - 2.4Hz

- 状态：`current`
- 首次确认：`2026-04-05`
- 最近复核：`2026-04-05`
- 数据范围：`工况1` 到 `工况20`
- 代码口径：
  - `src/current.data_loading`
  - `src/try/018_structural_fundamental_frequency_scan/`
- 证据入口：
  - `outputs/try/018_structural_fundamental_frequency_scan/case_frequency_summary.csv`
  - `outputs/try/018_structural_fundamental_frequency_scan/window_frequency_candidates.csv`
  - `outputs/try/018_structural_fundamental_frequency_scan/structural_frequency_overview.png`

结论：

- 在 `0.5Hz - 8.0Hz` 搜索范围内，按 `5s / 2.5s` 切窗，并屏蔽 `1x - 4x` 转频附近 `±0.2Hz` 后，`20` 个工况中有 `12` 个工况的应变侧候选频率中位数集中在约 `2.2Hz - 2.4Hz`
- 其余工况的应变侧候选频率中位数上移到约 `2.8Hz - 3.6Hz`，说明候选基频提取仍会受到工况状态、边界段差异或高阶结构响应影响
- 这说明“结构基频识别”当前更适合先走“物理启发式候选频率提取 + 稳定性验证”路线，而不是直接进入监督学习
- 下一步优先级应放在：
  - 稳态窗口筛选
  - 多传感器一致性评分
  - 与转频、倍频及边界段的进一步去耦
