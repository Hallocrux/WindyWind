# 工况误差模式聚类探索（2026-04-07）

- 状态：`current`
- 首次确认：`2026-04-07`
- 最近复核：`2026-04-07`
- 数据范围：带标签工况 `19` 个
- 代码口径：
  - `src/try/031_case_error_mode_clustering/`
- 证据入口：
  - `outputs/try/031_case_error_mode_clustering/case_error_profile_table.csv`
  - `outputs/try/031_case_error_mode_clustering/case_error_embedding.csv`
  - `outputs/try/031_case_error_mode_clustering/cluster_summary.csv`
  - `outputs/try/031_case_error_mode_clustering/error_mode_pca_scatter.png`
  - `outputs/try/031_case_error_mode_clustering/error_mode_heatmap.png`
  - `outputs/try/031_case_error_mode_clustering/summary.md`

## 1. 目标

在不重训模型的前提下，验证“高误差工况并不是同一种出错机制”：

- 构建 `per-case error profile table`
- 做低维投影与聚类
- 与第一层机制簇做对照

## 2. 方法口径

- 误差画像来源：
  - 风速窗长误差：`2s / 4s / 5s / 8s`
  - RPM 细窗长误差：`2.0s - 5.0s`
  - 边界差异特征：`003`
- 第一层机制簇只作为对照标签，不进入聚类
- 聚类方法：
  - 标准化后做 `AgglomerativeClustering`
  - 在 `k=2/3/4` 中按 silhouette 选最优

## 3. 当前结果

### 3.1 [2026-04-07] 当前误差画像更支持 `2` 类结构

- 状态：`current`
- 首次确认：`2026-04-07`
- 最近复核：`2026-04-07`

当前结果：

- 选定簇数：`2`
- silhouette score：`0.5604`
- PCA explained variance：
  - `PC1 = 53.03%`
  - `PC2 = 15.73%`

误差簇分配：

- `error cluster 0`
  - `3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,19,20`
- `error cluster 1`
  - `1,18`

### 3.2 [2026-04-07] `工况1` 与 `工况18` 形成了一个单独的特殊误差模式小簇

- 状态：`current`
- 首次确认：`2026-04-07`
- 最近复核：`2026-04-07`

`error cluster 1` 的主要特征是：

- `rpm_err_2_0s` 更高
- `rpm_err_3_5s` 更高
- `wind_long_short_delta` 更低

同时该簇的风速 `TinyTCN@5s` 平均误差更高：

- `error cluster 0`
  - `wind_loco_error_5s_mean = 0.2303`
- `error cluster 1`
  - `wind_loco_error_5s_mean = 0.9211`

这说明：

- `工况1` 与 `工况18` 的“难”，确实与其他高误差工况不同；
- 它们更像是一类“跨任务都对某些时间尺度不友好”的特殊小簇。

### 3.3 [2026-04-07] 第一层机制簇与第二层误差簇并不等价

- 状态：`current`
- 首次确认：`2026-04-07`
- 最近复核：`2026-04-07`

对照结果：

- `error cluster 1` 包含：
  - `工况1`
    - `mechanism cluster = 0`
  - `工况18`
    - `mechanism cluster = 1`

而 `工况17` 虽然风速误差也很高，但仍留在 `error cluster 0`。

这说明：

- 第一层的“数据机制簇”
- 与第二层的“模型出错模式簇”

不是同一个划分。

## 4. 当前判断

`2026-04-07` 的第二层探索支持以下判断：

- “工况不是同分布一团”已经不仅体现在原始机制特征，也体现在误差画像上；
- 但第一层机制簇和第二层误差簇并不等价；
- 当前最值得继续验证的下一步，是：
  - 在第一层机制簇下做“簇内训练 / 跨簇验证”
  - 或围绕 `工况1 / 18` 这类特殊误差簇单独做解释性检查。
