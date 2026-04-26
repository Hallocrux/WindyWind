# embedding top-k local prototype fusion quickcheck（2026-04-08）

- 状态：`current`
- 首次确认：`2026-04-08`
- 最近复核：`2026-04-08`
- 数据范围：
  - `final` 代表性 holdout：`工况1 / 3 / 17 / 18`
  - `added`：`工况21-24`
- 代码口径：
  - `src/try/056_embedding_topk_local_prototype_fusion/`
- 证据入口：
  - `outputs/try/056_embedding_topk_local_prototype_fusion/case_level_predictions.csv`
  - `outputs/try/056_embedding_topk_local_prototype_fusion/summary_by_domain.csv`
  - `outputs/try/056_embedding_topk_local_prototype_fusion/reference_neighbors.csv`
  - `outputs/try/056_embedding_topk_local_prototype_fusion/prototype_feature_table.csv`
  - `outputs/try/056_embedding_topk_local_prototype_fusion/models/`
  - `outputs/try/056_embedding_topk_local_prototype_fusion/summary.md`

## 1. 目标

验证一条更贴近“先做高维局部原型，再由小头输出修正”的最小版 quickcheck：

- 不再引入 `mechanism pool`；
- 参考工况池只由 `2s+8s` case embedding 的 `top-k` 决定；
- 不再直接对邻居 `wind residual` 做平均；
- 先构造 local embedding prototype，再用受约束小头从高维差异里输出 correction。

## 2. 方法口径

### 2.1 参考池

- 对每个目标工况：
  - 用 `TinyTCN 2s + 8s` concat 后的 case embedding 做 `top-k=4` 检索；
- 距离：
  - 在 fold 内对 concat embedding 各维做标准化后，使用欧氏距离；
- prototype：
  - 对 top-k 邻居按 `1 / distance` 做归一化加权均值，得到 `h_ref`。

### 2.2 小头输入

- `h`
- `h_ref`
- `h - h_ref`
- `|h - h_ref|`
- `base_pred`
- `reference_pool_size`
- `top1_embed_distance`
- `topk_embed_mean_distance`
- `topk_embed_std_distance`

### 2.3 小头

- `StandardScaler + RidgeCV`
- 监督目标：
  - `rpm_residual_oof`
- 输出方式：
  - 先预测 `atanh(clipped_residual / bound)`，再反变换
  - 得到 bounded correction
- 保留两个版本：
  - `w=1.0`
  - `w=0.5`

### 2.4 复用

- `2s / 8s` encoder：
  - 优先复用 `053` 已落盘 checkpoint
- 旧对照：
  - 直接复用 `052 / 053` 已落盘 case-level 预测

## 3. 当前结果

### 3.1 [2026-04-08] 最小版 embedding-prototype head 尚未超过 `052` 的固定 `2s+8s` concat residual

`focus_all` 对照：

- `052 | rpm_knn4 + embedding_residual_concat_2s_8s @ w=0.5`
  - `case_mae = 0.6006`
- `053 | rpm_knn4 + support_window_residual_avg_2s_8s @ w=0.5`
  - `case_mae = 0.6125`
- `056 | rpm_knn4 + embedding_topk_prototype_ridge @ w=0.5`
  - `case_mae = 0.7193`
- `056 | rpm_knn4 + embedding_topk_prototype_ridge`
  - `case_mae = 0.8526`

这说明：

- 把融合前移到高维 prototype 层这件事已经实现并跑通；
- 但当前最小版 head 还没有自然优于更简单的 case-level `2s+8s` concat residual；
- 当前证据还不支持把 `056` 直接升级成默认候选。

### 3.2 [2026-04-08] `056` 在部分 `final` hard case 上有正信号，但 added 整体退化明显

`final_focus` 对照：

- `052 | rpm_knn4 + embedding_residual_concat_2s_8s @ w=0.5`
  - `case_mae = 1.0159`
- `056 | rpm_knn4 + embedding_topk_prototype_ridge @ w=0.5`
  - `case_mae = 1.0872`
- `056 | rpm_knn4 + embedding_topk_prototype_ridge`
  - `case_mae = 1.1171`

代表工况：

- `工况3`
  - `rpm_knn4 = 1.1322`
  - `056 | w=1.0 = 0.7208`
- `工况17`
  - `rpm_knn4 = 1.5987`
  - `056 | w=1.0 = 1.1421`

但 `added_focus`：

- `rpm_knn4 = 0.2293`
- `056 | w=0.5 = 0.3514`
- `056 | w=1.0 = 0.5882`

这说明：

- prototype-conditioned head 不是完全没学到结构；
- 它在部分 `final` hard case 上确实能给出比 `rpm_knn4` 更强的修复；
- 但当前这套高维映射对 `added` 还不够稳，整体收益被外部域退化拖住了。

### 3.3 [2026-04-08] 当前高维 prototype head 的主要风险不是“找错邻居”，而是“小样本高维回归”仍不够保守

代表工况：

- `工况21`
  - `rpm_knn4 = 0.3799`
  - `056 | w=1.0 = 0.2312`
- `工况22`
  - `rpm_knn4 = 0.2658`
  - `056 | w=0.5 = 0.3691`
  - `056 | w=1.0 = 0.4725`
- `工况24`
  - `rpm_knn4 = 0.2294`
  - `056 | w=0.5 = 0.2123`
  - `056 | w=1.0 = 0.6539`

这说明：

- `embedding top-k` 参考池本身并没有完全崩坏；
- `056` 对 `工况21`、`工况24` 这种样本仍能给出局部修正；
- 但在当前样本量下，直接让高维 `[h, h_ref, delta, |delta|]` 驱动一个 ridge head，仍然容易把 correction 放大到对 added 不安全的程度；
- 当前更像是“prototype idea 有信号，但 head 约束方式还不够好”。

## 4. 当前判断

截至 `2026-04-08`，这轮 quickcheck 支持下面的表达：

- 去掉 `mechanism pool` 之后，参考池定义已经回到更符合原意的 `embedding top-k`；
- “局部 reference set -> local prototype -> constrained correction” 这条路线是可实现、可复现的；
- 但当前最小版 `ridge` 高维融合头还没有压住外部域退化；
- 因此这条线下一步更值得优先修的不是：
  - 把 head 再做大
- 而是：
  - 进一步收紧 correction 约束；
  - 减少高维原始输入的自由度；
  - 或把 prototype 差异先变成更保守的 bucket / low-rank signal，再决定 correction。

## 5. 一句话版结论

截至 `2026-04-08`，`056` 已经把参考池成功改回 `embedding top-k`，并完成了“高维 prototype 融合前移”的首个最小版验证；但当前 `ridge` 小头还没有超过固定 `2s+8s` concat residual，尤其 added 外部域退化仍明显，因此下一步更值得优先修的是 correction 约束与表征压缩方式，而不是重新引入 mechanism pool。
