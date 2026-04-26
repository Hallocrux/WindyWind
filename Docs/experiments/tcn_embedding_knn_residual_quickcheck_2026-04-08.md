# TinyTCN embedding kNN residual quickcheck（2026-04-08）

- 状态：`current`
- 首次确认：`2026-04-08`
- 最近复核：`2026-04-08`
- 数据范围：
  - `data/final/` 的带标签工况 `1, 3-20`
  - `data/added/` 的带标签工况 `21-24`
- 代码口径：
  - `src/try/051_tcn_embedding_knn_residual/`
- 证据入口：
  - `outputs/try/051_tcn_embedding_knn_residual/case_level_predictions.csv`
  - `outputs/try/051_tcn_embedding_knn_residual/summary_by_domain.csv`
  - `outputs/try/051_tcn_embedding_knn_residual/nearest_neighbors.csv`
  - `outputs/try/051_tcn_embedding_knn_residual/summary.md`

## 1. 目标

验证一个更局部的思路是否成立：

- 不再强制让 `TinyTCN` 直接回归风速；
- 先用 `TinyTCN@2s` 学窗口级表征；
- 再把 case 级 embedding 用于：
  - `embedding -> kNN -> wind`
  - `rpm_knn4 + embedding -> kNN residual`

本轮问题是：

- `TinyTCN` 在跨域时既然已经表现出明显差异，这些差异能不能转化成“局部近邻修正信号”。

## 2. 方法口径

- 训练池：
  - `final` 带标签工况 `19` 个
  - `added` 带标签工况 `4` 个
  - 总计 `23` 个带标签工况
- 评估：
  - 统一 case-level `LOOCV`
- 时序模型：
  - `TinyTCN@2s`
- embedding 定义：
  - 取 `TinyTCN` 最后卷积块经全局平均池化后的 `32` 维 case 级均值向量
- 近邻：
  - `k = 4`
  - 距离加权平均
- 对照变体：
  - `rpm_knn4`
  - `TinyTCN direct`
  - `embedding_knn4`
  - `rpm_knn4 + embedding_residual_knn4`
  - `rpm_knn4 + 0.5 * embedding_residual_knn4`

## 3. 当前结果

### 3.1 [2026-04-08] 统一池整体最优仍然是 `rpm_knn4`

- `all_labeled | rpm_knn4`
  - `case_mae = 0.3634`
- `all_labeled | rpm_knn4 + embedding_residual_knn4 @ w=0.5`
  - `case_mae = 0.3700`
- `all_labeled | rpm_knn4 + embedding_residual_knn4`
  - `case_mae = 0.4029`
- `all_labeled | TinyTCN direct`
  - `case_mae = 0.5564`
- `all_labeled | embedding_knn4`
  - `case_mae = 0.5791`

这说明：

- 当前 `TCN embedding` 还不能直接替代 `rpm_knn4` 成为统一池主干；
- “把 `TCN` 变成 embedding 再直接做 kNN 回归”并没有自然优于原始 `TinyTCN direct`；
- 当前更有保留价值的是“embedding 只做小修正”，而不是“embedding 直接管最终预测”。

### 3.2 [2026-04-08] `embedding residual` 在 added 子集上有明显正信号

- `added | rpm_knn4`
  - `case_mae = 0.2293`
- `added | rpm_knn4 + embedding_residual_knn4`
  - `case_mae = 0.1728`
- `added | rpm_knn4 + embedding_residual_knn4 @ w=0.5`
  - `case_mae = 0.1698`

这说明：

- `TinyTCN` 学到的表征里，确实包含了一部分与 added 偏移相关的局部修正信号；
- 这部分信号直接拿来回归风速不稳，但叠加在强解析基线后可以起作用；
- 当前更像“embedding 可作为 added correction feature”，而不是“embedding 本身就是更好的主预测器”。

### 3.3 [2026-04-08] `embedding residual` 仍然会伤害 final 子集

- `final | rpm_knn4`
  - `case_mae = 0.3917`
- `final | rpm_knn4 + embedding_residual_knn4 @ w=0.5`
  - `case_mae = 0.4121`
- `final | rpm_knn4 + embedding_residual_knn4`
  - `case_mae = 0.4514`

这说明：

- 当前 residual 修正信号还不能默认在全域开启；
- 它比“直接学风速”稳，但仍没有解决 `final` 保护问题；
- 因此这条线的更合理角色不是“统一默认修正器”，而是“需要额外门控的 optional correction”。

### 3.4 [2026-04-08] `工况22` 是这条线最明显的受益点之一

`工况22` 对照：

- `rpm_knn4`
  - `pred = 3.1342`
  - `abs_error = 0.2658`
- `rpm_knn4 + embedding_residual_knn4`
  - `pred = 3.4307`
  - `abs_error = 0.0307`
- `TinyTCN direct`
  - `pred = 4.5191`
  - `abs_error = 1.1191`
- `embedding_knn4`
  - `pred = 4.5948`
  - `abs_error = 1.1948`

这说明：

- 当前 embedding 最有价值的地方，不是“自己直接预测”，而是“帮助修正 hardest added case”；
- `工况22` 这类异常点上，`base + local residual` 比直接 learned 预测更合理。

## 4. 当前判断

截至 `2026-04-08`，这轮 quickcheck 支持下面的表达：

- `TinyTCN` 的表征确实包含局部相似性信号；
- 但当前最优用法不是 `embedding -> direct kNN wind`；
- 更有前景的写法是：
  - `base = rpm_knn4`
  - `optional correction = embedding residual`
- 不过这条 correction 仍然主要对 added 有利，对 final 还不够安全。

因此下一步若继续追这条线，更值得做的不是：

- 直接扩大 embedding kNN 本身

而是：

- 给 `embedding residual` 加显式门控；
- 或只在“疑似 added-like / 异常机制点”上允许开启。

## 5. 一句话版结论

截至 `2026-04-08`，`TinyTCN embedding` 已经证明“局部近邻修正”这条路有 added 方向的正信号，但它还没有强到可以取代 `rpm_knn4`；当前更合理的角色是“解析主干上的可选局部修正器”，而不是统一主模型。
