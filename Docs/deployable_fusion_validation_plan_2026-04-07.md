# 可部署融合路线验证计划（2026-04-07）

- 状态：`current`
- 首次确认：`2026-04-07`
- 最近复核：`2026-04-07`
- 数据范围：
  - 主训练域：`data/final/` 的带标签工况
  - 外部验证域：`data/added/` 的 `工况21-24`
- 关联代码口径：
  - 多尺度主干：`src/try/026_tinytcn_priority1_quickcheck/`
  - RPM 回归：`src/try/019_tinytcn_rpm_regression/`
  - RPM 细窗长：`src/try/024_tinytcn_rpm_fine_window_scan/`
  - 中频应变：`src/try/039_strain_transfer_band_scan/`
  - 解析 / learned 混合：`src/try/041_rpm_vs_learned_midband_check/`
  - 多 seed 复核：`src/try/042_rpm_learned_midband_multiseed_stability_check/`

## 1. 计划目标

本文档用于把 `2026-04-07` 已经分散得到的探索结论，收敛为一条可部署、可复核、可决定是否升级主线的执行计划。

计划要回答的最终问题是：

- 如果部署时不能依赖真实 `rpm`，当前表格主线应该如何组合：
  - `2s/8s` 多尺度风速主干
  - `pred_rpm` 派生的解析辅助支线
  - `strain(3.0-6.0Hz)` learned 中频支线
- 最终应该形成：
  - 一个统一主线
  - 还是“旧域主线 + added 增强线”的双轨方案

## 2. 当前已知前提

### 2.1 [2026-04-07] 真实 RPM 不能作为部署时的必要输入

- 状态：`current`
- 首次确认：`2026-04-07`
- 最近复核：`2026-04-07`

当前 `041/042` 里表现最强的 `rpm_knn4 + learned midband` 方案，使用的是 manifest 中的真实 `rpm`。

这意味着：

- 该结果当前只能视作“可利用 RPM 信息时的上界”；
- 不能直接把真实 `rpm` 依赖带入未来默认主线；
- 后续任何想保留 `rpm` 路线的方案，都必须改写成：
  - `sensor -> pred_rpm -> rpm_to_wind`
  - 并按这一可部署链路重新复核。

### 2.2 [2026-04-07] `2s/8s` 多尺度风速模型仍是当前旧域最强主干证据

- 状态：`current`
- 首次确认：`2026-04-07`
- 最近复核：`2026-04-07`

截至 `2026-04-07`，对 `data/final/` 旧域最强、且不依赖真实 `rpm` 的主干证据仍来自 `026`：

- `TinyTCN_multiscale_late_fusion_2s_8s`
  - `case_mae = 0.1685`
  - `case_rmse = 0.2522`

因此后续统一融合路线中，默认主干应优先写成：

- `A = 2s/8s` 多尺度风速模型

而不是直接让 `midband` 或 `rpm` 支线替代主干。

### 2.3 [2026-04-07] `3.0-6.0Hz` 中频应变已证明有独立增益，但当前证据主要来自 added

- 状态：`current`
- 首次确认：`2026-04-07`
- 最近复核：`2026-04-07`

截至 `2026-04-07`：

- `039/040` 已把 added 方向上最可迁移的应变频带收敛到约 `3.0-6.0Hz`；
- `041/042` 已证明 `rpm + learned midband` 的混合路线在 added 上稳定成立；
- 但还没有证据表明：
  - 该中频支线在 `final` 旧域 `LOCO` 上也稳定有益；
  - 它可以直接升级为旧域默认主干。

因此当前更合理的角色设定是：

- `B = strain(3.0-6.0Hz)` learned 辅助支线

而不是直接把它写成全局主干。

### 2.4 [2026-04-07] `rpm + learned midband` 的固定权重当前应优先参考 `0.3`

- 状态：`current`
- 首次确认：`2026-04-07`
- 最近复核：`2026-04-07`

`042` 的多随机种子复核显示：

- `fusion @ w=0.3`
  - `case_mae mean = 0.1627`
  - `best_seed_count = 7 / 10`
- `fusion @ w=0.5`
  - `case_mae mean = 0.1822`
  - `best_seed_count = 3 / 10`

这说明：

- 混合路线本身是稳定成立的；
- 但 `041` 单次 seed 里的 `w=0.5` 不应继续作为默认固定权重；
- 如果后续仍保留解析 + learned 混合，默认固定权重应先从更保守的 `0.3` 邻域起步。

## 3. 执行总原则

### 3.1 统一评估口径

后续所有 try 必须同时输出以下两类评估，不再只看一边：

- `final` 旧域：带标签工况 `LOCO`
- `added` 外部域：`工况21-24`

### 3.2 严格区分“上界实验”和“可部署实验”

后续涉及 `rpm` 的路线必须显式区分：

- `true_rpm` 上界参考
- `pred_rpm` 可部署版本

除非是专门做“上界分析”的 try，否则默认只汇报可部署版本。

### 3.3 优先做晚融合，再做结构级联合

在统一主线还未验证前，不优先做大一统端到端网络。

建议顺序：

- 先做工况级 / case 级晚融合
- 再做受约束权重搜索
- 最后才考虑把多个分支做成一个联合网络

### 3.4 主干优先、辅助保守

在统一融合矩阵里，默认应满足：

- `A = 2s/8s` 为主干
- `B = midband` 为辅助
- `C = pred_rpm` 为辅助

初始权重搜索不应让 `B` 或 `C` 主导整体预测。

### 3.5 所有“主线升级”都必须同时满足双域条件

一个候选若只在 `added` 上好，而在 `final` 上退化，则不得直接升级为全局默认主线。

这种情况只能进入：

- `added` 专项增强线
- 或双轨方案候选

## 4. 具体 Try 拆分计划

以下 try 编号按 `042` 之后顺延，默认从 `043` 开始。

### 4.1 `043_pred_rpm_deployability_check`

**目标**

- 验证“真实 `rpm` 不可用”前提下，`rpm` 支线是否仍有保留价值；
- 把当前 `true_rpm -> rpm_knn4` 上界，转换成可部署的 `pred_rpm -> rpm_to_wind` 口径。

**核心问题**

- `pred_rpm` 替代 `true_rpm` 后，added 上的性能掉多少；
- `rpm_knn4` 是否对 `pred_rpm` 误差过于敏感；
- 是否存在比 `knn4` 更适合“预测 rpm 输入”的平滑映射。

**建议变体**

- `true_rpm -> rpm_knn4`
- `pred_rpm_3.0s -> rpm_knn4`
- `pred_rpm_3.0s -> rpm_linear`
- `pred_rpm_3.0s -> ridge_rpm_to_wind`
- 如有余力：
  - `pred_rpm_2.0s -> rpm_linear`
  - `pred_rpm_5.0s -> rpm_linear`

**输入与复用**

- rpm 预测模型优先复用：
  - `src/try/019_tinytcn_rpm_regression/`
  - `src/try/024_tinytcn_rpm_fine_window_scan/`
- 默认优先使用 `3.0s` rpm 模型作为部署候选

**输出**

- `rpm_case_level_predictions.csv`
- `rpm_to_wind_summary.csv`
- `deployable_vs_true_rpm_gap.csv`
- `summary.md`

**通过条件**

- 至少存在一个 `pred_rpm -> wind` 方案，在 `added` 上仍显著优于纯 learned 中频支线；
- 且其相对 `true_rpm` 版本的退化不至于抹掉全部融合收益。

**失败回退**

- 如果 `pred_rpm` 路线退化过大，则后续统一主线中：
  - `C = pred_rpm` 只保留为研究支线
  - 不进入默认候选融合矩阵

### 4.2 `043_3_fft_midband_fusion_replay`

**目标**

- 在不重跑上游识别结果的前提下，先补齐最直接的 deployable 问题：
  - 用 FFT 支线替代 `true_rpm`；
  - 检查它能否继续与 `3.0-6.0Hz learned midband` 形成稳定融合；
  - 作为后续统一矩阵前的最短闭环验证。

**核心问题**

- `fft_rpm -> wind` 替代 `true_rpm -> wind` 后，added 融合收益还剩多少；
- FFT 替代版与 `true_rpm` 上界相比差多少；
- 当前更稳的 deployable 权重是 `0.3` 还是 `0.5`。

**建议变体**

- `fft_peak_1x_whole -> rpm_knn4`
- `hybrid_peak_1x_whole_window8_gate150 -> rpm_knn4`
- `window_peak_1x_conf_8s -> rpm_knn4`
- 每个 FFT 解析支线分别与 `TinyTCN all_channels midband` 做：
  - `w = 0.3 / 0.5 / 0.7`

**初始权重建议**

- `fft_branch:0.7 / learned:0.3`
- `fft_branch:0.5 / learned:0.5`
- `fft_branch:0.3 / learned:0.7`

**输出**

- `case_level_predictions.csv`
- `seed_summary.csv`
- `stability_overview.csv`
- `pairwise_comparison.csv`
- `summary.md`

**通过条件**

- 至少存在一个 FFT deployable 融合候选同时满足：
  - 明显优于 FFT 单独链；
  - 与 `true_rpm + learned midband` 上界只剩小幅差距。

**失败回退**

- 如果 FFT 替代后与 `true_rpm` 上界差距仍明显偏大，则暂不把 FFT 写入最终融合默认线，只保留为解析辅助支线。

### 4.3 `045_dual_domain_validation_matrix`

**目标**

- 对 `044` 的前 2-3 名候选做正式双域复核；
- 决定是走统一主线，还是走双轨方案。

**核心问题**

- added 增益是否以牺牲 final 为代价；
- 哪些 hardest case 在不同方案下显著受益或受损。

**固定评估桶**

- `final LOCO`
- `added 21-24`
- `hard cases`
  - `工况1`
  - `工况18`
  - `工况22`

**建议输出**

- `candidate_rank_final.csv`
- `candidate_rank_added.csv`
- `candidate_tradeoff_matrix.csv`
- `hard_case_breakdown.csv`
- `summary.md`

**通过条件**

- 若统一候选同时满足：
  - `final` 指标不劣于当前主干 `A`
  - `added` 明显优于当前非 rpm 主干
  - hardest case 无明显新增长尾
  - 则可进入“统一主线候选”

**失败回退**

- 若 added 提升与 final 退化无法兼得，则输出明确双轨结论：
  - `final`：保留 `A`
  - `added`：使用增强方案

### 4.4 `046_multiseed_final_candidate_check`

**目标**

- 对 `045` 选出的统一候选或双轨候选，做多随机种子复核；
- 避免主线升级建立在单个 seed 上。

**建议对象**

- 统一候选前 2 名
- 或双轨候选各自默认方案

**输出**

- `seed_summary.csv`
- `best_variant_by_seed.csv`
- `pairwise_comparison.csv`
- `summary.md`

**通过条件**

- 默认候选在多 seed 下：
  - 平均优于基线
  - 标准差可接受
  - 最优比例不低

**失败回退**

- 若新方案 seed 敏感明显高于现主线，则不得升级默认主线。

### 4.5 `047_joint_branch_architecture_smoke`

**目标**

- 仅在 `046` 通过后，探索是否有必要把晚融合升级成联合结构；
- 不再回答“有没有用”，只回答“能否减少部署复杂度并保持效果”。

**建议结构**

- 主干：
  - `2s/8s` 多尺度分支
- 辅助子分支：
  - `midband 3.0-6.0Hz`
- 可选第二子头：
  - `pred_rpm` 辅助任务或辅助输入

**输出**

- `joint_vs_late_fusion_summary.csv`
- `case_level_comparison.csv`
- `summary.md`

**通过条件**

- 联合模型必须同时满足：
  - 不差于晚融合
  - 推理链路更简单或更统一

**失败回退**

- 若联合结构没有稳定正信号，继续保留晚融合作为工程默认方案。

## 5. 各 Try 的依赖关系

按顺序建议如下：

1. `043_pred_rpm_deployability_check`
2. `043_3_fft_midband_fusion_replay`
3. `045_dual_domain_validation_matrix`
4. `046_multiseed_final_candidate_check`
5. `047_joint_branch_architecture_smoke`

其中：

- `043` 未通过时，优先转入 `043_3` 的 FFT 替代验证，不再默认带 `C = pred_rpm`
- `045` 未通过双域统一条件时，`046` 改做双轨候选稳定性复核
- `046` 未通过时，不进入 `047`

## 6. 每个阶段的验收问题

### 6.1 `043` 验收问题

- 预测 rpm 是否能替代真实 rpm 进入后续主线；
- 解析支线是否仍值得保留。

### 6.2 `043_3` 验收问题

- FFT 是否已经可以替代 `true_rpm`，进入 added deployable 融合线。

### 6.3 `045` 验收问题

- 当前项目是否存在一条统一主线；
- 若不存在，双轨方案边界在哪里。

### 6.4 `046` 验收问题

- 候选方案是否稳，而不是单次偶然最优。

### 6.5 `047` 验收问题

- 是否值得从晚融合升级到真正联合架构。

## 7. 当前推荐的默认工程策略

在 `043-047` 完成前，`2026-04-07` 的临时默认策略建议是：

- `final` 旧域默认参考：
  - `2s/8s` 多尺度风速主干
- `added` 增强默认参考：
  - `rpm_knn4 + TinyTCN all_channels midband(3.0-6.0Hz)`
  - 固定权重默认参考 `0.3`
- 但该 added 增强线当前仍属于：
  - 外部域增强候选
  - 不是全局默认主线

## 7.1 [2026-04-07] 若双域结果继续分化，当前更推荐 route / gate，而不是硬做统一主线

- 状态：`current`
- 首次确认：`2026-04-07`
- 最近复核：`2026-04-07`
- 证据入口：
  - `Docs/route_gate_strategy_2026-04-07.md`

当前更值得优先验证的是：

- 不直接学习“这是 `final` 还是 `added`”；
- 而是学习：
  - 当前样本更适合走哪条路线
  - 或当前样本的增强权重应当是多少

当前更推荐的最小路线是：

- 先做 case 级 `route / gate quickcheck`
- gate 模型优先参考：
  - `LogisticRegression`
  - `LightGBM / XGBoost / CatBoost`
- 暂不优先继续扩大 `TCN` 门控网络

这条线的目标不是证明“域分类是否存在”，而是回答：

- 能否在压住 `final` 退化的同时，保住 `added` 上的增强收益

## 8. 文档与落盘要求

后续每个 try 除了代码与输出外，都应同步补一份 `Docs/experiments/` 文档，并至少写清楚：

- 目标
- 方法口径
- 变体矩阵
- 当前结果
- 当前判断
- 如被后续 try 替代，补 `替代关系`

如果某个 try 的结论已经稳定进入当前项目认知，再把最终仍有效的部分回写到 `PROJECT.md`。
