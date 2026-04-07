# added 反常表现域诊断（2026-04-07）

- 状态：`current`
- 首次确认：`2026-04-07`
- 最近复核：`2026-04-07`
- 数据范围：
  - 主训练域：`data/final/` 的 `工况1-20`
  - added 外部工况：`data/added/` 的 `工况21-24`
- 代码口径：
  - `src/try/036_added_domain_diagnosis/`
- 证据入口：
  - `outputs/try/036_added_domain_diagnosis/added_mechanism_diagnostics.csv`
  - `outputs/try/036_added_domain_diagnosis/nearest_final_cases.csv`
  - `outputs/try/036_added_domain_diagnosis/added_baseline_comparison.csv`
  - `outputs/try/036_added_domain_diagnosis/added_feature_outliers.csv`
  - `outputs/try/036_added_domain_diagnosis/case22_reference_comparison.csv`
  - `outputs/try/036_added_domain_diagnosis/projection_scatter.png`
  - `outputs/try/036_added_domain_diagnosis/case22_spectrum_comparison.png`
  - `outputs/try/036_added_domain_diagnosis/summary.md`

## 1. 目标

在 `034/035` 已确认 `added 21-24` 存在系统性高估后，进一步区分：

- 这是输入通道口径变化导致的假异常；
- 还是 `added` 本身已经脱离当前 `final` 主训练域；
- 以及 `工况22` 是否应被单独视作异常机制点。

## 2. 方法口径

- 不重新训练 `TinyTCN`，直接复用：
  - `034` 的 clean pool 外部预测
  - `035` 的 full final pool 外部预测
- 清洗与切窗继续复用：
  - `src/current.data_loading`
  - `src/current.features`
- 机制特征继续复用：
  - `030` 的 per-case mechanism feature 口径
- 诊断内容包括：
  - `final` 与 `final + added` 的共有通道一致性检查
  - `added` 投影到 `final` 机制特征空间
  - `added -> final` 最近邻搜索
  - `rpm-only` 基线与 `TinyTCN` 外部预测对照
  - `工况22` 的异常特征排序与频谱对照

## 3. 当前结果

### 3.1 [2026-04-07] `added` 没有改变当前共有输入通道口径

- 状态：`current`
- 首次确认：`2026-04-07`
- 最近复核：`2026-04-07`

共有通道检查结果：

- `final` 单独共有通道数：`20`
- `final + added` 共有通道数：`20`
- 当前没有发现：
  - 因 `added` 引入而丢失的共有列
  - 或因文件头差异导致的输入列收缩

这说明：

- `034/035` 的异常高估，不像是“把 added 拼进来以后输入通道被裁坏了”。

### 3.2 [2026-04-07] `工况22` 在机制特征空间里几乎是孤点

- 状态：`current`
- 首次确认：`2026-04-07`
- 最近复核：`2026-04-07`

最近邻结果：

- `工况21`
  - 最近邻：`工况19`
  - 距离：`5.0918`
- `工况22`
  - 最近邻：`工况1`
  - 距离：`25.4185`
- `工况23`
  - 最近邻：`工况13`
  - 距离：`3.7998`
- `工况24`
  - 最近邻：`工况19`
  - 距离：`3.9134`

这说明：

- `工况21 / 23 / 24` 虽然也偏离主域，但仍能在 `final` 中找到相对接近的机制邻居；
- `工况22` 与全部 `final` 工况的距离明显更大，更像一个需要单独解释的异常机制点。

### 3.3 [2026-04-07] `工况23` 更接近 `15-18` 型子域，而 `21/24` 更接近主簇

- 状态：`current`
- 首次确认：`2026-04-07`
- 最近复核：`2026-04-07`

按 `030` 的机制簇质心距离：

- `工况21`
  - 更近 `cluster 0`
- `工况22`
  - 更近 `cluster 0`
- `工况23`
  - 更近 `cluster 1`
- `工况24`
  - 更近 `cluster 0`

这说明：

- `added` 不是一个单一子域；
- `工况23` 更像已知高能量 / 低应变幅值子域的延伸；
- `工况21 / 24` 更像主簇附近的偏移样本；
- `工况22` 则不是“普通 cluster 0 低风速样本”，而是额外叠加了更强的频谱异常。

### 3.4 [2026-04-07] `工况22` 的应变频带结构与主数据显著不同

- 状态：`current`
- 首次确认：`2026-04-07`
- 最近复核：`2026-04-07`

`工况22` 的主要异常特征：

- `strain_low_ratio_median = 0.9052`
  - `z = +21.1748`
- `strain_mid_ratio_median = 0.0931`
  - `z = -20.8667`
- `acc_peak_freq_median = 14.8267`
  - `z = +3.0258`
- `strain_rms_median = 1405.3307`
  - `z = -2.3752`
- `acc_mid_ratio_median = 0.0526`
  - `z = -2.2665`

这说明：

- `final` 主数据大多数工况以 `2-5Hz` 的应变中频占主导；
- `工况22` 却几乎被 `0-2Hz` 应变低频主导；
- 因此它更像“频谱机制改变”的点，而不只是“幅值偏一点”的普通边界样本。

### 3.5 [2026-04-07] `rpm-only` 基线对 `21/23/24` 明显优于当前 TinyTCN 外部预测

- 状态：`current`
- 首次确认：`2026-04-07`
- 最近复核：`2026-04-07`

对照结果：

- `工况21`
  - 真值：`4.4`
  - `rpm_linear = 4.1370`
  - `clean_5s = 6.5895`
  - `full_5s = 7.8197`
- `工况22`
  - 真值：`3.4`
  - `rpm_linear = 2.3728`
  - `rpm_knn3 = 2.8637`
  - `clean_5s = 6.5977`
  - `full_5s = 8.2063`
- `工况23`
  - 真值：`6.0`
  - `rpm_linear = 6.1858`
  - `clean_5s = 6.7246`
  - `full_5s = 7.4192`
- `工况24`
  - 真值：`5.0`
  - `rpm_linear = 5.1614`
  - `clean_5s = 6.5793`
  - `full_5s = 7.9458`

这说明：

- 对 `工况21 / 23 / 24`，单独使用 RPM 的简单基线已经比当前 TinyTCN 外推更接近标签；
- `工况22` 即使在 RPM 基线下也有误差，但仍明显好于当前外部高估；
- 当前 added 反常的主问题更像“信号机制外推失败”，不是“RPM 标签整体不可信”。

## 4. 当前判断

`2026-04-07` 的这轮诊断支持以下判断：

- `added` 反常高估的主解释仍应优先放在“域偏移 / 机制偏移”，而不是输入列收缩；
- `工况22` 当前应被视作最高优先级的单独异常机制点；
- `工况23` 更像 `15-18` 型子域的延伸；
- `工况21 / 24` 则更像主簇附近但幅值口径不同的偏移样本；
- 如果下一步要继续追因，优先顺序应是：
  - 先复核 `工况22` 的标签链路与原始采集背景；
  - 再做“只用 RPM / 只用应变 / 只用加速度”的外部对照；
  - 最后再考虑 added 是否需要独立建模或做域适配。
