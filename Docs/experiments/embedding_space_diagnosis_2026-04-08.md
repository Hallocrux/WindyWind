# embedding space diagnosis（2026-04-08）

- 状态：`current`
- 首次确认：`2026-04-08`
- 最近复核：`2026-04-08`
- 数据范围：
  - `final`：`工况1-20`
  - `added`：`工况21-24`
- 代码口径：
  - `src/try/057_embedding_space_diagnosis/`
- 证据入口：
  - `outputs/try/057_embedding_space_diagnosis/embedding_case_table.csv`
  - `outputs/try/057_embedding_space_diagnosis/embedding_pca_coords.csv`
  - `outputs/try/057_embedding_space_diagnosis/pairwise_distance_matrix.csv`
  - `outputs/try/057_embedding_space_diagnosis/knn_neighbors.csv`
  - `outputs/try/057_embedding_space_diagnosis/hubness_counts.csv`
  - `outputs/try/057_embedding_space_diagnosis/summary.md`
  - `outputs/try/057_embedding_space_diagnosis/plots/`

## 1. 目标

验证一个更基础的问题：

- `embedding_concat` 是否具备可解释的局部结构；
- 它能否作为默认候选检索空间；
- `added` 工况在统一 embedding 空间中是“有桥接”还是“完全断裂”。

本轮不做风速预测，只做统一坐标系下的 embedding 诊断。

## 2. 方法口径

### 2.1 统一坐标系

- 不再复用 holdout fold 各自的坐标系；
- 直接用全部 `23` 个带标签工况：
  - `final` 带标签 `19` 个
  - `added` 带标签 `4` 个
- 各训练一套统一的：
  - `TinyTCN@2s`
  - `TinyTCN@8s`
- 对所有 `24` 个工况导出：
  - `embedding_2s`
  - `embedding_8s`
  - `embedding_concat`

### 2.2 诊断输出

- `PCA 2D`
- pairwise distance heatmap
- `top-4` 邻居表
- hubness 统计
- PCA 平面上的 `top-1` 邻居边图

## 3. 当前结果

### 3.1 [2026-04-08] 统一 embedding 空间在 `final` 域内表现出明显的局部结构

统一空间摘要：

- 导出工况数：`24`
- 训练工况数：`23`
- `top-4`
- PCA explained variance：
  - `PC1 = 62.18%`
  - `PC2 = 25.59%`
- overall same-domain neighbor rate：`83.33%`
- `final_labeled` same-domain neighbor rate：`94.74%`

代表邻域：

- `工况4 -> 5 / 9 / 7 / 11`
- `工况5 -> 4 / 9 / 7 / 3`
- `工况6 -> 11 / 10 / 19 / 8`
- `工况13 -> 14 / 12 / 20 / 8`
- `工况14 -> 13 / 12 / 8 / 20`
- `工况15 -> 16 / 17 / 23 / 18`
- `工况16 -> 15 / 17 / 23 / 18`

这说明：

- 统一 `embedding_concat` 并不是无结构的高维空间；
- 在 `final` 域内已经形成了较清楚的局部簇；
- 多组近邻关系与现有直觉一致，尤其：
  - `13 / 14`
  - `15 / 16 / 17`
  - `4 / 5 / 7 / 9`

### 3.2 [2026-04-08] `added` 在统一 embedding 空间中不是完全断裂，而是分成两个局部邻域

`added` 邻域摘要：

- `工况21 -> 24 / 22 / 7 / 10`
- `工况22 -> 21 / 24 / 1 / 3`
- `工况23 -> 16 / 15 / 17 / 24`
- `工况24 -> 21 / 22 / 10 / 23`

`added` same-domain neighbor rate：

- `50.00%`

这说明：

- `added` 并没有在统一 embedding 空间中完全漂散成孤点；
- 当前更像是分成两条局部线：
  - `21 / 22 / 24`
  - `23` 靠近 `15 / 16 / 17`
- 这支持“embedding space 具有可用的 added 检索结构”；
- 但 `added` 还没有形成一个完全封闭、完全稳的子簇，跨域邻居仍然会进入 top-k。

### 3.3 [2026-04-08] 当前 embedding 检索空间存在 hubness，但还没有塌缩到极少数点

Top hub cases：

- `工况8`
  - `selected_count = 8`
- `工况11`
  - `selected_count = 7`
- `工况10`
  - `selected_count = 6`
- `工况7`
  - `selected_count = 6`
- `工况5`
  - `selected_count = 5`

这说明：

- 统一 embedding 空间里确实存在“更容易被检索到”的中枢工况；
- 但当前 hubness 还没有塌缩到 `1-2` 个点统治全图；
- 对默认检索空间来说，这更像“需要监控的风险”，而不是立即否定 embedding 检索的证据。

## 4. 当前判断

截至 `2026-04-08`，这轮诊断更支持下面的表达：

- `embedding_concat` 已经足够形成有意义的局部结构；
- 它比此前的 `mechanism pool` 更符合“默认检索空间”的角色；
- 当前证据支持把它保留为默认检索空间候选；
- 但 added 方向还不够完全闭合，后续继续做 prototype / correction 时，仍需要：
  - 邻域稳定性复核；
  - hubness 监控；
  - added 样本的保守约束。

## 5. 一句话版结论

截至 `2026-04-08`，统一坐标系下的 `embedding_concat` 已经表现出明确的局部簇结构，尤其 `final` 域内结构很清楚，`added` 也不是完全断裂而是形成了两组局部邻域；这支持把 `embedding_concat` 保留为默认检索空间候选，但 added 外部域仍需要保守约束与后续稳定性复核。
