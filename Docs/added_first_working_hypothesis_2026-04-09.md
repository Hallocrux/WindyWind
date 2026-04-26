# added-first 工作假设（2026-04-09）

- 状态：`historical`
- 性质：`working_hypothesis`
- 首次提出：`2026-04-09`
- 最近复核：`2026-04-09`
- 替代关系：
  - `2026-04-09` 的阶段性工作假设；
  - 子项目收尾后的最终默认结论见 `Docs/added_first_strategy_and_alignment_2026-04-09.md`
- 数据范围：
  - `data/added/` 的 `工况21-24`
  - `data/added2/` 的 `工况25-30`
- 代码口径：
  - `src/try/069_added2_embedding_pca_projection/`
  - `src/try/070_added_added2_pca_line_regression/`
  - `src/try/071_external_embedding_regression_quickcheck/`
  - `src/try/072_external_embedding_topk_loocv/`
- 证据入口：
  - `outputs/try/069_added2_embedding_pca_projection/`
  - `outputs/try/070_added_added2_pca_line_regression/`
  - `outputs/try/071_external_embedding_regression_quickcheck/`
  - `outputs/try/072_external_embedding_topk_loocv/`

## 1. 文档目的

本文档只记录截至 `2026-04-09` 的一条**当前工作假设**：

- 在后续一小段探索周期内，可以把“优先优化 `added / added2` 外部域表现”作为主目标；
- 在这条假设下，允许某些 try 暂时不以“同时保住 `final`”作为第一目标；
- 但这**不等于**已经证明：
  - `final` 不再重要；
  - 统一主线已经切换；
  - 当前新增路线已经稳定超过旧修复模型。

换句话说，本文档是 `2026-04-09` 当天“如何组织 try 优先级”的记录，不是最终定版结论。

## 2. 当前工作假设

### 2.1 [2026-04-09] 当前可以把 `added-first` 作为 try 空间的阶段性主假设

截至 `2026-04-09`，更合理的阶段性组织方式是：

- 对新的探索线，主优化目标优先参考：
  - `added -> added2` 外推；
  - `added + added2` 外部域内部 `LOOCV`；
- `final` 伤害在这个阶段仍然记录，但可以先降级为次级约束；
- 若某条路线只在 `final` 上表现更好、但无法改善 `added / added2`，则当前优先级应后移。

这个假设的核心不是“宣布 unified 主线已经切换”，而是：

- 当前外部域证据已经足够强，值得先围绕 `added / added2` 形成更清晰的局部方法；
- 等外部域路线稳定后，再决定是否反推统一主线。

## 3. 支持这条工作假设的证据

### 3.1 [2026-04-09] `added` 与 `added2` 在统一 embedding 空间里形成同一外部域族

`069` 的 `TinyTCN 2s+8s` PCA 投影显示：

- `added` 与 `added2` 整体上和旧 `final` 主簇分离；
- `added2` 虽然不是单一团块，但仍与 `added` 共享明显的外部域结构；
- 这支持“先在外部域内部做建模或修正”，而不是继续把主要目标放在旧 `final LOCO`。

注意：

- 这条证据支持“外部域优先值得做”；
- 但不支持“`added` 与 `added2` 完全同构”或“存在一条单一回归直线即可解释全部风速变化”。

### 3.2 [2026-04-09] 单独沿 PCA 线回归不够，但高维 embedding 本身有外部域预测力

`070` 显示：

- 把 `added + added2` 压到 PCA 主轴后做一维回归，效果有限；
- 但二维 PCA 回归仍比“单轴回归”更好，说明压缩前的高维 embedding 里仍有有效信息。

这支持：

- 后续继续用高维 embedding 做建模；
- 不建议把“PCA 图上的一条线”直接升级为最终回归形式。

### 3.3 [2026-04-09] `rpm_knn4 + 高维 embedding residual` 在 `added -> added2` 协议下出现强正信号

`071` 的 `added -> added2` 结果显示：

- `rpm_knn4 + embedding residual ridge`
  - `case_mae = 0.6161`
- `rpm_knn4`
  - `case_mae = 1.2903`
- `embedding_ridge`
  - `case_mae = 1.2572`

这说明：

- 如果只问“能否用 `added` 学到的外部域结构去改善 `added2`”，当前已经出现明显正信号；
- 外部域优化这条线不再只是概念性假设，而是已经有可运行的候选方法。

### 3.4 [2026-04-09] `top-k` 选择对 `added2` 局部修正有效，但还不够稳定

`072` 的 external `LOOCV` 结果显示：

- 全体 `added + added2` 统一看时，最好的是：
  - `embedding_knn4`
  - `case_mae = 0.7455`
- 但只看 `added2` 子集时，最好的是：
  - `rpm_knn4 + topk4 residual mean`
  - `case_mae = 0.5520`

这说明：

- “added-first” 并不等于“已经找到统一最优结构”；
- 但它支持把 `added2` 当成主优化对象继续细化局部修正器。

## 4. 当前不足与风险

### 4.1 [2026-04-09] 证据还不足以宣布 unified 主线已经切换

当前不能直接下以下结论：

- `added-first` 已经正式取代 unified 主线；
- `final` 不再需要进入默认评估；
- 新 embedding residual 路线已经稳定超过所有旧修复模型。

主要原因：

- `added / added2` 样本仍极少；
- 不同 try 采用的协议不同：
  - `added -> added2`
  - `added + added2 external LOOCV`
  - `final + added + added2`
- 当前最佳结果还没有经过足够强的多随机种子和多切分复核。

### 4.2 [2026-04-09] 当前更适合把它理解为“研究组织方式变化”，而不是“工程默认结论变化”

截至 `2026-04-09`，更稳妥的表达是：

- 研究优先级已经明显偏向 `added / added2`；
- 但工程默认主线是否切换，仍需更严格验证。

## 5. 当前建议的工作规则

### 5.1 [2026-04-09] 对后续 try 的默认优先级建议

在下一阶段 try 里，优先级建议改成：

1. 首先回答：
   - `added -> added2` 是否还能继续降低误差；
   - `added2` 上哪些局部分支最值得单独修复；
2. 其次记录：
   - 该方法对 `added + added2` 外部域内部 `LOOCV` 是否稳定；
3. 最后再看：
   - `final` 是否同步保住；
   - 是否存在可以回推 unified 主线的机会。

### 5.2 [2026-04-09] 当前最合适的语气

截至 `2026-04-09`，推荐使用以下语气描述当前阶段：

- “当前工作假设更支持先把 `added / added2` 做通，再讨论 unified 主线。”
- “当前允许在 try 空间里暂时接受一定的 `final` 伤害，以换取对外部域的更清晰建模。”
- “这是一条阶段性研究假设，不是已经定版的最终结论。”

## 6. 下一步何时可以升级为更强结论

只有在至少满足以下条件后，才更适合把这条假设升级成更强判断：

1. `added -> added2` 上的新路线经过多随机种子复核仍领先；
2. `added + added2` external `LOOCV` 上的新路线仍保持优势；
3. 新路线对 `final` 的伤害边界已经被定量复核；
4. 最终能明确回答：
   - 是“外部域专用分支”更合理；
   - 还是“新的统一主线”已经形成。
