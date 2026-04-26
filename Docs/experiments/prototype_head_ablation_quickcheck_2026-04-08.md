# prototype head ablation quickcheck（2026-04-08）

- 状态：`current`
- 首次确认：`2026-04-08`
- 最近复核：`2026-04-08`
- 数据范围：
  - `final` 代表性 holdout：`工况1 / 3 / 17 / 18`
  - `added`：`工况21-24`
- 代码口径：
  - `src/try/058_prototype_head_ablation_quickcheck/`
- 证据入口：
  - `outputs/try/058_prototype_head_ablation_quickcheck/case_level_predictions.csv`
  - `outputs/try/058_prototype_head_ablation_quickcheck/summary_by_domain.csv`
  - `outputs/try/058_prototype_head_ablation_quickcheck/reference_neighbors.csv`
  - `outputs/try/058_prototype_head_ablation_quickcheck/prototype_feature_table.csv`
  - `outputs/try/058_prototype_head_ablation_quickcheck/delta_pca_feature_table.csv`
  - `outputs/try/058_prototype_head_ablation_quickcheck/models/`
  - `outputs/try/058_prototype_head_ablation_quickcheck/summary.md`

## 1. 目标

在 `056` 已经显示“embedding top-k + local prototype` 可实现，但 full input head 在 added 上不够稳”的前提下，做两个最小版 ablation，回答：

- added 退化是否主要来自 head 自由度过大；
- 如果只保留 `delta` 相关输入，是否会更稳；
- 如果 correction 只从低维 `delta` 里输出，added 退化是否还会明显下降。

## 2. 方法口径

### 2.1 参考池

- 完全复用 `056`：
  - `embedding top-k`
  - `top-k = 4`
  - `2s+8s` concat case embedding
  - 局部 prototype 由 `1 / distance` 加权平均构造

### 2.2 057a 风格：delta-only ridge

- 输入只保留：
  - `delta`
  - `|delta|`
  - `base_pred`
  - `reference_pool_size`
  - `top1_embed_distance`
  - `topk_embed_mean_distance`
  - `topk_embed_std_distance`
- 输出：
  - bounded correction
  - `w = 0.5`

### 2.3 057b 风格：low-rank delta ridge

- 先对训练 fold 的 `delta` 做 `PCA`
  - `n_components = 6`
- correction 只从低维 `delta_pca` 输出
- 不再输入：
  - `h`
  - `h_ref`
  - `|delta|`
  - `base_pred`
  - `dist_stats`
- 输出：
  - bounded correction
  - `w = 0.5`

## 3. 当前结果

### 3.1 [2026-04-08] 两个最小版都显著缓解了 `056 full input` 在 added 上的退化

`added_focus` 对照：

- `rpm_knn4`
  - `case_mae = 0.2293`
- `056 | embedding_topk_prototype_ridge @ w=0.5`
  - `case_mae = 0.3514`
- `058 | delta_only_prototype_ridge @ w=0.5`
  - `case_mae = 0.2612`
- `058 | lowrank_delta_prototype_ridge @ w=0.5`
  - `case_mae = 0.3186`

这说明：

- 一旦拿掉 `h / h_ref` 绝对坐标和 full input，自由度降低后，added 退化确实明显收敛；
- `delta-only` 收敛效果最明显；
- 这支持“056 的主要问题确实在 head 自由度，而不是 embedding top-k 检索本身”。

### 3.2 [2026-04-08] `delta-only` 比 `low-rank delta` 更像“稳住 added 的最小修复”

代表工况：

- `工况24`
  - `056 full input = 0.2123`
  - `delta-only = 0.0852`
  - `low-rank delta = 0.1217`
- `工况23`
  - `056 full input = 0.5185`
  - `delta-only = 0.1130`
  - `low-rank delta = 0.0170`
- `工况22`
  - `056 full input = 0.3691`
  - `delta-only = 0.4560`
  - `low-rank delta = 0.6961`

这说明：

- `delta-only` 整体上更稳地压住了 added 的过修正；
- `low-rank delta` 在 `工况23` 上很强，但在 `工况22` 上仍然不稳；
- 当前 added 修复更像“保留全部 delta 但限制 head 输入范围”比“直接压成极低维”更有效。

### 3.3 [2026-04-08] 降自由度虽然缓解了 added 退化，但并没有把 `final` 同时推强

`final_focus` 对照：

- `052 | rpm_knn4 + embedding_residual_concat_2s_8s @ w=0.5`
  - `case_mae = 1.0159`
- `056 | embedding_topk_prototype_ridge @ w=0.5`
  - `case_mae = 1.0872`
- `058 | delta_only_prototype_ridge @ w=0.5`
  - `case_mae = 1.1828`
- `058 | lowrank_delta_prototype_ridge @ w=0.5`
  - `case_mae = 1.1249`

这说明：

- 当前最小版 ablation 更像“控制副作用”的实验，而不是“直接刷出更强统一主线”；
- 它们验证了问题所在，但还没有把 `final + added` 两边同时兼顾好；
- 因此下一步若继续追这条线，更值得做的是：
  - 先围绕保守 correction / gate 继续设计
  - 而不是把 delta-only 或 low-rank delta 直接升级成默认模型。

## 4. 当前判断

截至 `2026-04-08`，这轮 ablation 更支持下面的表达：

- `056` 的 added 退化，主要确实来自 head 自由度过大；
- 把小头收紧到 `delta-only` 后，added 退化明显缓解；
- 但单纯降自由度还不足以让 unified `focus_all` 直接超过 `052` 的固定 `2s+8s` residual；
- 这意味着后续更值得优先追的方向是：
  - `delta-only + 更保守 gate`
  - 或 `delta-only + bucket / trigger`
  - 而不是回到 full input prototype head。

## 5. 一句话版结论

截至 `2026-04-08`，`058` 说明 added 退化的主要矛盾确实在 prototype head 自由度过大；把输入收紧到 `delta-only` 后，added 误差已从 `0.3514` 明显回落到 `0.2612`，这支持后续继续沿“embedding top-k 检索 + 更保守的 delta correction”推进，而不是回到 full input head。
