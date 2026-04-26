# contrastive condition embedding quickcheck（2026-04-12）

- 状态：`current`
- 首次确认：`2026-04-12`
- 最近复核：`2026-04-12`
- 数据范围：
  - encoder 对比预训练：`final` 带标签工况 `1, 3-20` + `added` 带标签工况 `21-24`
  - residual 训练：`data/added/` 的带标签工况 `21-24`
  - 测试：`data/added2/` 的带标签工况 `25-30`
- 代码口径：
  - `src/try/086_contrastive_condition_embedding/`
- 证据入口：
  - `outputs/try/086_contrastive_condition_embedding/summary_by_protocol.csv`
  - `outputs/try/086_contrastive_condition_embedding/case_level_predictions.csv`
  - `outputs/try/086_contrastive_condition_embedding/embedding_case_table.csv`
  - `outputs/try/086_contrastive_condition_embedding/knn_neighbors.csv`
  - `outputs/try/086_contrastive_condition_embedding/summary.md`
- 模型资产：
  - `outputs/try/086_contrastive_condition_embedding/models/checkpoints/contrastive_2s.pt`
  - `outputs/try/086_contrastive_condition_embedding/models/checkpoints/contrastive_2s_norm.npz`
  - `outputs/try/086_contrastive_condition_embedding/models/checkpoints/contrastive_2s.json`
  - `outputs/try/086_contrastive_condition_embedding/models/checkpoints/contrastive_8s.pt`
  - `outputs/try/086_contrastive_condition_embedding/models/checkpoints/contrastive_8s_norm.npz`
  - `outputs/try/086_contrastive_condition_embedding/models/checkpoints/contrastive_8s.json`

## 1. 目标

验证一个不直接依赖风速监督的工况表征路线：

- 用 `case_id` 组织正负样本；
- 让同一工况的窗口 embedding 接近；
- 让不同工况的窗口 embedding 远离；
- 再把导出的 case embedding 接入 `071` 同类的 `rpm_knn4 + residual ridge` 评估。

本轮主要回答：

- contrastive condition embedding 能否替换 `071` 使用的风速监督 TinyTCN embedding；
- 在 `added -> added2` 主口径下，是否超过 `071 | rpm_knn4 + embedding residual ridge` 的 `case_mae = 0.6161`。

## 2. 方法口径

- encoder：
  - TinyTCN blocks
  - Temporal Pyramid Pooling `[1, 2, 4]`
  - projection head 输出 `64` 维 normalized window embedding
- 窗长：
  - `2s`
  - `8s`
- case embedding：
  - 每个窗长下，对同一工况的 window embedding 求均值；
  - 再把 `2s + 8s` 拼接为 `128` 维 case embedding。
- 训练目标：
  - supervised contrastive loss；
  - 标签只使用 `case_id`；
  - encoder 训练阶段不使用 `wind_speed`。
- 下游评估：
  - `rpm_knn4`
  - `contrastive_embedding_ridge`
  - `rpm_knn4 + contrastive_embedding_residual_ridge`

## 3. 当前结果

### 3.1 [2026-04-12] Contrastive embedding residual 已超过 `rpm_knn4`，但没有超过 `071`

`added_to_added2` 结果：

- `rpm_knn4 + contrastive_embedding_residual_ridge`
  - `case_mae = 0.9264`
  - `case_rmse = 1.0956`
  - `max_abs_error = 1.9502`
- `rpm_knn4`
  - `case_mae = 1.2903`
  - `case_rmse = 1.5511`
  - `max_abs_error = 2.9911`
- `contrastive_embedding_ridge`
  - `case_mae = 1.6242`
  - `case_rmse = 1.7412`
  - `max_abs_error = 2.7796`

对照当前 `071`：

- `071 | rpm_knn4 + embedding residual ridge`
  - `case_mae = 0.6161`

这说明：

- 不用风速监督训练出来的工况表征，已经能给 `rpm_knn4` 提供 residual 修正信号；
- 但当前 `086` 最小版仍明显弱于 `071`；
- 因此 `086` 当前不能替代 `071` 成为 added-first 默认最佳模型。

### 3.2 [2026-04-12] 当前修正主要改善高风速端，但低风速端出现过冲

逐工况对照显示：

- `工况25`
  - `rpm_knn4 abs_error = 2.9911`
  - `rpm_knn4 + contrastive residual abs_error = 1.9502`
- `工况26`
  - `rpm_knn4 abs_error = 1.7651`
  - `rpm_knn4 + contrastive residual abs_error = 0.6933`
- `工况27`
  - `rpm_knn4 abs_error = 1.0240`
  - `rpm_knn4 + contrastive residual abs_error = 0.0376`
- `工况28`
  - `rpm_knn4 abs_error = 0.4901`
  - `rpm_knn4 + contrastive residual abs_error = 0.9266`
- `工况29`
  - `rpm_knn4 abs_error = 0.7685`
  - `rpm_knn4 + contrastive residual abs_error = 1.2537`
- `工况30`
  - `rpm_knn4 abs_error = 0.7032`
  - `rpm_knn4 + contrastive residual abs_error = 0.6972`

这说明：

- 当前 contrastive residual 对 `工况25-27` 有明显正向修正；
- 但对 `工况28-29` 出现过冲；
- 该路线若继续推进，下一步更需要 gate / trust 约束，而不是直接扩大 residual 强度。

## 4. 当前判断

截至 `2026-04-12`，这轮 quickcheck 支持下面的表达：

- “不使用风速监督、只按工况相似性训练 encoder”这条线是可运行的；
- 它在 `added_to_added2` 上已经优于单独 `rpm_knn4`；
- 但它没有超过 `071` 的风速监督 embedding residual；
- 因此当前更合理的角色是研究型表征候选，不是默认主线升级。

若继续追这条线，优先方向应是：

- 加入 residual gate，避免 `工况28-29` 的过冲；
- 对 `2s / 8s` 分别做 ablation，确认哪个窗长贡献了高风速端修正；
- 比较 `case_id` 对比学习与 `rpm bucket + case_id` 混合正负样本定义。
