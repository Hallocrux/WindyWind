# added-first 策略与认知对齐（2026-04-09）

- 状态：`current`
- 性质：`strategy_note`
- 首次确认：`2026-04-09`
- 最近复核：`2026-04-09`
- 数据范围：
  - `data/added/` 的 `工况21-24`
  - `data/added2/` 的 `工况25-30`
- 代码口径：
  - `src/try/071_external_embedding_regression_quickcheck/`
  - `src/try/072_external_embedding_topk_loocv/`
  - `src/try/073_external_052_embedding_residual_replay/`
  - `src/try/074_external_059_delta_gate_replay/`
  - `src/try/075_from_others_rule_model_added_first_eval/`
  - `src/try/076_from_others_multitask_inference_added_series/`
  - `src/try/077_true_fft_rpm_blend_quickcheck/`
  - `src/try/078_repo_fft_rpm_blend_quickcheck/`
  - `src/try/079_repo_fft_sideinfo_in_071_residual/`
  - `src/try/080_tabular_linear_baselines_added_first_eval/`
- 证据入口：
  - `outputs/try/071_external_embedding_regression_quickcheck/`
  - `outputs/try/072_external_embedding_topk_loocv/`
  - `outputs/try/073_external_052_embedding_residual_replay/`
  - `outputs/try/074_external_059_delta_gate_replay/`
  - `outputs/try/075_from_others_rule_model_added_first_eval/`
  - `outputs/try/076_from_others_multitask_inference_added_series/`
  - `outputs/try/077_true_fft_rpm_blend_quickcheck/`
  - `outputs/try/078_repo_fft_rpm_blend_quickcheck/`
  - `outputs/try/079_repo_fft_sideinfo_in_071_residual/`
  - `outputs/try/080_tabular_linear_baselines_added_first_eval/`

## 1. 文档目的

本文档记录截至 `2026-04-09` 当前已经对齐的三类内容：

- 当前 added-first 方向的统一评估口径；
- 当前 added-first 口径下的默认最佳模型；
- 当前已经验证过、但不应继续作为主候选推进的旧路线与外部路线。

本文档不是历史实验流水账；历史上下文与早期工作假设仍以：

- `Docs/added_first_working_hypothesis_2026-04-09.md`

为准。

## 2. 当前统一认知

### 2.1 [2026-04-09] 当前更合理的问题定义是“先把 added/added2 做到可用”

截至 `2026-04-09`，当前更合理的研究组织方式不是继续优先追求：

- `final + added + added2` 的统一主线最优；

而是优先回答：

- 只利用 `added(21-24)` 的已知结构，能否稳定改善 `added2(25-30)`；
- 哪些 external-only 修正器最值得保留。

这意味着：

- `final` 当前不再作为新 try 的主排序依据；
- `final` 当前保留为次级参考约束；
- 当前主目标是 external family 的可用性，而不是统一主线的漂亮性。

### 2.2 [2026-04-09] 当前更支持“系统发生了持久变化”，而不是“只是 final/added 的暂时漂移”

截至 `2026-04-09`，当前更合理的认知是：

- `added` 与 `added2` 在统一 embedding 空间里形成了同一类外部域族；
- 这支持“added 系列与旧 final 主簇已经分开”的判断；
- 当前工作不应再默认把 `added` 理解成“仍然属于 final 主域、只需轻微修补”的情况。

这条认知来自：

- `069`
- `070`
- `071`
- `072`

的联合证据。

### 2.3 [2026-04-09] `rpm` 是已提供的工况参数，允许直接依赖

截至 `2026-04-09`，此前把某些 `true_rpm` 路线仅视为“deployable 上界参考”的理解已被撤回。

当前统一口径改为：

- `rpm` 是已提供参数；
- 允许直接进入默认模型设计；
- 因此 `rpm_knn4`、`rpm + learned branch` 等路线都应被视为正式可用候选，而不是仅作理论上界。

## 3. 当前统一评估口径

### 3.1 [2026-04-09] 当前 added-first 主口径应优先参考 `added -> added2`

截至 `2026-04-09`，若必须给出 added-first 的单一主口径，当前应优先采用：

- 训练域：`added(工况21-24)`
- 测试域：`added2(工况25-30)`
- 指标：case-level `MAE`

当前推荐命名为：

- `added_to_added2`

这条口径当前优先级高于：

- `added + added2 external LOOCV`

原因不是后者无用，而是前者更直接对应：

- “旧 external 域能否推到新 external 域”

这一当前主问题。

### 3.2 [2026-04-09] `added + added2 external LOOCV` 当前保留为次级复核口径

截至 `2026-04-09`，下列口径仍然有保留价值：

- `added + added2 external LOOCV`

它当前主要用于回答：

- 一个方法在 external family 内部是否整体稳定；
- 某些局部修正是否只是对 `added2` 某一小组工况偶然有效。

但当前它不应取代 `added_to_added2` 成为单一主口径。

## 4. 当前 added-first 默认最佳模型

### 4.1 [2026-04-09] 当前主口径下的默认最佳模型是 `071 | rpm_knn4 + embedding residual ridge`

截至 `2026-04-09`，在 `added_to_added2` 主口径下，当前默认最佳模型为：

- `rpm_knn4 + embedding residual ridge`

对应结果：

- `case_mae = 0.6161`

对照：

- `rpm_knn4`
  - `case_mae = 1.2903`
- `embedding_ridge`
  - `case_mae = 1.2572`
- `embedding_knn4`
  - `case_mae = 1.9126`

这说明：

- 当前 added-first 最有保留价值的不是 direct learned 主干；
- 而是 `rpm` 主干上叠加 external embedding residual 的结构。

## 5. 当前已重判但不应升级为主候选的路线

### 5.1 [2026-04-09] `052` replay 在 external-first 口径下只有弱信号，不应回升为默认主候选

`073` 对 `052` 的 replay 显示：

- `added_to_added2`
  - `052_embedding_residual_knn4_concat_2s_8s_w0.5`
  - `case_mae = 1.2337`
- `rpm_knn4`
  - `case_mae = 1.2903`

这说明：

- 旧 `052` 在主口径下只显示出弱改善；
- 它没有接近当前 `071` 的水平；
- 因此可以保留为次级参考，但不应回升为当前默认主候选。

### 5.2 [2026-04-09] `059` replay 在 external-first 口径下不应继续作为主方向

`074` 对 `059` 的 replay 显示：

- `added_to_added2`
  - `059_delta_only_trigger_rule_cv`
  - `case_mae = 1.2903`
- `rpm_knn4`
  - `case_mae = 1.2903`

这说明：

- 旧 `059` 的 trigger/gate 思路在当前主口径下没有形成新的可迁移收益；
- 它不应继续作为当前 external-first 主方向。

### 5.3 [2026-04-09] `from_others/1` 当前不应进入默认候选矩阵

`075` 显示：

- `added_to_added2`
  - `from_others_rule_model_v1`
  - `case_mae = 1.3245`

这说明：

- `from_others/1` 的整体风速预测器当前弱于 `rpm_knn4`；
- 其“FFT + 物理先验”的思路可保留；
- 但该实现本身不应进入当前默认候选矩阵。

### 5.4 [2026-04-09] `from_others/2` 现成权重在当前数据上退化为近常数预测，不应继续使用

`076` 显示：

- `added`
  - `case_mae = 0.8159`
- `added2`
  - `case_mae = 2.0167`

并且：

- 各工况几乎都预测为约 `5.0318`

这说明：

- `from_others/2` 的现成权重与当前 added 系列数据口径明显不匹配；
- 当前不应继续作为可用候选。

### 5.5 [2026-04-09] 补测后，`G6` 与 `Ridge + VIB_FT_RPM` 也不构成 added-first 主候选

`080` 在当前协议下补测显示：

- `added_to_added2`
  - `071 | rpm_knn4 + embedding residual ridge`
  - `case_mae = 0.6161`
- `added_to_added2`
  - `rpm_knn4`
  - `case_mae = 1.2903`
- `added_to_added2`
  - `Ridge + VIB_FT_RPM`
  - `case_mae = 1.4522`
- `added_to_added2`
  - `TabularReference_G6_Ridge`
  - `case_mae = 1.9618`

这说明：

- `G6` 此前虽然没有在 added-first 协议下单独补测；
- 但补测后可以明确排除它作为当前默认主候选的可能性；
- 正式表格主线里的 `Ridge + VIB_FT_RPM` 也没有在主口径上接近 `071`；
- 因此当前不存在一个被漏掉的强线性表格挑战者。

## 6. 当前对 FFT / RPM 中间量的统一认知

### 6.1 [2026-04-09] `rpm` 仍然是有价值的中间量

截至 `2026-04-09`，当前统一认知不是“放弃 rpm”，而是：

- `rpm` 仍然是当前风速回归里最有价值的中间量之一；
- 当前 external-first 最稳主干依旧是 `rpm_knn4`；
- 新信息应优先考虑作为 `rpm` 主干的修正或辅助，而不是直接推翻 `rpm` 主干。

### 6.2 [2026-04-09] `from_others/1` 的固定 FFT 求 RPM 口径不应直接并入当前模型

`077` 显示：

- `added_to_added2`
  - `rpm_true_knn4 = 1.2903`
  - `rpm_mix05_knn4 = 1.3719`
  - `rpm_fft_knn4 = 1.4695`

这说明：

- `from_others/1` 那套固定 FFT RPM 估计不适合作为当前默认 `rpm` 替代；
- 固定 `0.5` 混合也没有形成正信号。

### 6.3 [2026-04-09] 仓库内 FFT RPM 口径只显示出极弱边际正信号，固定 `0.5` 混合不成立

`078` 显示：

- `added_to_added2`
  - `rpm_repo_fft_knn4 = 1.2872`
  - `rpm_true_knn4 = 1.2903`
  - `rpm_repo_mix05_knn4 = 1.2916`

这说明：

- 仓库内 `043_1` 的 FFT RPM 口径明显优于 `from_others/1` 那版；
- 但固定 `0.5` 混合没有形成正信号；
- 当前能保留的判断只有：
  - `repo_fft_rpm` 作为辅助中间量存在极弱边际价值；
  - 但它不值得直接替换 `true_rpm` 主干，也不值得做固定权重混合。

## 7. 当前默认策略

### 7.1 [2026-04-09] 当前默认主线继续保持 `071`，不升级为固定 `fft_rpm` 混合版

截至 `2026-04-09`，当前默认主线策略应保持：

- 主干：`rpm_knn4`
- 修正：`embedding residual ridge`

不应直接升级为：

- `true_rpm + fft_rpm @ 0.5`
- `repo_fft_rpm` 替代 `true_rpm`

### 7.2 [2026-04-09] 仓库 FFT side-info 注入 `071` residual 分支已完成验证，但没有形成升级版

`079` 已按最小版完成验证：

- 保留 `071` 主干不变；
- 只在 residual ridge 输入里加入：
  - `repo_fft_rpm`
  - `repo_delta_rpm`
  - `repo_abs_delta_rpm`
  - `repo_fft_confidence`
  - `repo_fft_source`

`079` 的结果显示：

- `added_to_added2`
  - `071 | rpm_knn4 + embedding residual ridge`
  - `case_mae = 0.6161`
- `added_to_added2`
  - `079 | rpm_knn4 + embedding + repo FFT side-info residual ridge`
  - `case_mae = 0.7292`
- `added + added2 external LOOCV`
  - `071`
  - `case_mae = 0.8451`
- `added + added2 external LOOCV`
  - `079`
  - `case_mae = 1.0584`

这说明：

- FFT side-info 直接并入 `071` residual ridge 后，仍然优于 `rpm_knn4`；
- 但它没有超过 `071`；
- 并且改善只表现为 `3/6` case 分裂，不满足“稳定升级”的要求；
- 因此这条线到这里应停止，不再继续展开 `delta-only` follow-up。

### 7.3 [2026-04-09] added-first 子项目的最终默认结论就是 `071`

截至 `2026-04-09`，当前 added-first 子项目收尾后的默认结论应固定为：

- 默认最佳模型：
  - `071 | rpm_knn4 + embedding residual ridge`
- 主口径：
  - `added_to_added2`
- 当前结果：
  - `case_mae = 0.6161`

这意味着当前不再继续作为主候选推进的方向包括：

- 固定 `0.5` rpm 混合；
- 直接把 FFT 路线抬为主干；
- 继续沿 `from_others/1` 的风速集成器直接扩写。
- 把 repo FFT side-info 直接并入 `071` residual ridge。

若后续重新开启 added-first 子项目，应把 `071` 视为默认对照位，而不是重新从 `052 / 059 / 075 / 077 / 078 / 079` 这些支线开始。

## 8. 当前难点工况认知

### 8.1 [2026-04-09] 当前 hardest cases 主要集中在 `23 / 24 / 25 / 27`

截至 `2026-04-09`，当前已识别的 hardest cases 可粗分为：

- `工况23`
  - 主要问题更像邻域整体偏高；
- `工况24`
  - `rpm` 信号与 embedding 信号出现冲突；
- `工况25`
  - 高风速端缺少足够支撑；
- `工况27`
  - 边界样本，容易 correction 过冲。

这说明：

- 当前若继续追 added-first 改善，重点不应再是平均意义上的泛化；
- 而应优先围绕这些 hardest cases 设计局部修正和可信度控制。
