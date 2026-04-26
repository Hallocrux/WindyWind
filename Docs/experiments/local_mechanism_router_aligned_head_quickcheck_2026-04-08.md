# local mechanism router aligned head quickcheck（2026-04-08）

- 状态：`current`
- 首次确认：`2026-04-08`
- 最近复核：`2026-04-08`
- 数据范围：
  - `final` 代表性 holdout：`工况1 / 3 / 17 / 18`
  - `added`：`工况21-24`
- 代码口径：
  - `src/try/055_local_mechanism_router_aligned_head_quickcheck/`
- 证据入口：
  - `outputs/try/055_local_mechanism_router_aligned_head_quickcheck/case_level_predictions.csv`
  - `outputs/try/055_local_mechanism_router_aligned_head_quickcheck/summary_by_domain.csv`
  - `outputs/try/055_local_mechanism_router_aligned_head_quickcheck/router_case_neighbors.csv`
  - `outputs/try/055_local_mechanism_router_aligned_head_quickcheck/aligned_feature_table.csv`
  - `outputs/try/055_local_mechanism_router_aligned_head_quickcheck/models/`
  - `outputs/try/055_local_mechanism_router_aligned_head_quickcheck/summary.md`

## 1. 目标

验证一个更贴近“先限机制区域、再做表示对齐、最后用小头输出”的最小版 quickcheck：

- 不再直接在全池 embedding 上做 residual kNN；
- 先用 case-level mechanism feature 限制局部参考池；
- 再比较两条路线：
  - 局部机制池内直接 residual kNN
  - 局部机制池内做 aligned representation，再由 bounded head 输出 correction

## 2. 复用策略

- `2s / 8s` encoder：
  - 优先复用 `053` 已落盘 fold checkpoint
- `gate / mechanism` 特征：
  - 直接复用 `047` 的 `gate_feature_table.csv`
- 旧对照：
  - 直接复用 `052 / 053` 已落盘 case-level 预测
- 新训练模型：
  - 只训练本轮新增的 bounded ridge head
  - 每个 holdout fold 的 head 全部落盘到 `models/`

## 3. 方法口径

### 3.1 局部机制路由

- 对每个目标工况：
  - 先在训练池里按 mechanism feature 距离选 `top-6` 局部参考池
- mechanism feature 复用：
  - `missing_ratio_in_common_cols`
  - `edge_removed_ratio`
  - `strain_low_ratio_median`
  - `strain_mid_ratio_median`
  - `strain_low_over_mid`
  - `strain_rms_median`
  - `acc_energy_median`
  - `acc_peak_freq_median`
  - `strain_acc_rms_ratio`
  - `hour_sin`
  - `hour_cos`

### 3.2 局部 residual 对照

- 在上述局部机制池内：
  - 用 `2s+8s` case embedding 做 `k=4` residual kNN
  - 目标仍为 `rpm_residual_oof`
  - 最终仍保守使用 `w=0.5`

### 3.3 aligned head

- 在局部机制池内：
  - 用 embedding 距离做 support attention
  - 取 support centroid 与目标 case 表征差异
- 特征包括：
  - `target / support / delta PCA`
  - top-k embedding distance 统计
  - support `base_pred / rpm` gap
  - target mechanism 与 support mechanism centroid 的差分
- 小头：
  - `StandardScaler + RidgeCV`
  - 先预测 `atanh(clipped_residual / bound)`，再反变换
  - 输出 bounded correction
  - 同时保留 `w=1.0` 与 `w=0.5` 两个版本

## 4. 当前结果

### 4.1 [2026-04-08] 全局最优仍然是 `052` 的固定 `2s+8s` concat residual

`focus_all` 对照：

- `052 | rpm_knn4 + embedding_residual_concat_2s_8s @ w=0.5`
  - `case_mae = 0.6006`
- `053 | rpm_knn4 + support_window_residual_avg_2s_8s @ w=0.5`
  - `case_mae = 0.6125`
- `055 | rpm_knn4 + local_mechanism_residual_knn4_concat_2s_8s @ w=0.5`
  - `case_mae = 0.6252`
- `055 | rpm_knn4 + local_mechanism_aligned_tanh_ridge_pca6 @ w=0.5`
  - `case_mae = 0.7146`
- `055 | rpm_knn4 + local_mechanism_aligned_tanh_ridge_pca6`
  - `case_mae = 0.8527`

这说明：

- 当前“先做局部机制路由”还没有把整体结果推到超过 `052`；
- `aligned head` 在当前样本规模下仍然不稳；
- 当前最稳的默认候选仍然是 `052` 的固定 `2s+8s` concat residual。

### 4.2 [2026-04-08] 先做局部机制收窄本身不是完全无效

`final_focus` 对照：

- `052 | rpm_knn4 + embedding_residual_concat_2s_8s @ w=0.5`
  - `case_mae = 1.0159`
- `055 | rpm_knn4 + local_mechanism_residual_knn4_concat_2s_8s @ w=0.5`
  - `case_mae = 1.0064`

代表工况：

- `工况3`
  - `rpm_knn4 = 1.1322`
  - `local_mechanism_residual = 1.0457`
- `工况17`
  - `rpm_knn4 = 1.5987`
  - `local_mechanism_residual = 1.4416`

这说明：

- “先限到局部机制区域再做修正”这个问题定义本身不是错的；
- 仅仅把参考池从全池缩到机制局部池，已经能在部分 `final` hard case 上给出一点正信号；
- 但这点收益当前还不够大，也没有稳定迁移到 `added`。

### 4.3 [2026-04-08] `aligned head` 具备局部单点修复能力，但整体仍被路由与样本量限制

代表工况：

- `工况21`
  - `rpm_knn4 = 0.3799`
  - `local_mechanism_aligned_tanh_ridge_pca6 = 0.0013`
- `工况3`
  - `rpm_knn4 = 1.1322`
  - `local_mechanism_aligned_tanh_ridge_pca6 = 0.7557`

但整体：

- `final_focus | aligned_head_full = 1.2425`
- `added_focus | aligned_head_full = 0.4628`

这说明：

- 对齐表征 + 小头并不是完全没有学到东西；
- 它能在少数工况上做出强修复；
- 但当前整体表现仍被“局部机制池不稳 + 样本过少”拖住，不能当作默认升级路线。

### 4.4 [2026-04-08] `工况22` 暴露了当前机制路由的核心问题：added 外部样本在机制空间上仍未被校准

`aligned_feature_table.csv` 中，`工况22` 的关键量为：

- `local_mechanism_mean_distance = 188.0048`
- `top1_embed_distance = 11.1490`
- `support_base_gap = -1.4937`

`router_case_neighbors.csv` 显示：

- `工况22` 的局部机制池被路由到：
  - `工况10 / 6 / 18 / 14 / 5 / 20`
- 对应 mechanism distance 全都在约 `1454-1456`

这说明：

- 当前 `047` 那套 mechanism feature 在旧域内部能工作，但拿来直接做 `added` 外部路由还不够稳；
- `工况22` 在现有 mechanism 坐标系里几乎是“无邻居”的状态；
- 这会直接把后续 residual / aligned head 都带偏。

## 5. 当前判断

截至 `2026-04-08`，这轮 quickcheck 更支持下面的表达：

- “先限制到局部相似机制区域”这一步值得保留；
- 但当前最弱的环节已经不再是“有没有局部表示”，而是“现有 mechanism feature 是否能把外部域样本稳地路由到对的局部池”；
- 在局部池本身不稳的前提下，继续扩大 aligned head 自由度并不会自然变好。

因此下一步更值得做的不是：

- 继续加大 case-level 小头容量

而是：

- 先修 mechanism 空间的外部域校准；
- 或把路由写成更保守的“blocklist / allowlist”；
- 或对 added-like case 先做机制 bucket，再在 bucket 内做 aligned head。

## 6. 一句话版结论

截至 `2026-04-08`，`055` 说明“先做局部机制路由”本身有弱正信号，但当前 mechanism 空间对 added 外部样本还不够稳，导致 `aligned head` 整体上仍未超过固定 `2s+8s` concat residual；下一步更值得优先修的是局部路由的机制口径，而不是继续放大小头。
