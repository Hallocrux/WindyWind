# 风速回归实验总报告（2026-04-08）

- 状态：`current`
- 首次确认：`2026-04-08`
- 最近复核：`2026-04-08`
- 数据范围：
  - `final`：`工况1-20`
  - `added`：`工况21-24`
  - `added2`：`工况25-30`
- 代码口径：
  - 正式主线：`src/current/`
  - 探索主线：`src/try/001-064`
- 证据入口：
  - 项目状态卡：`PROJECT.md`
  - 表格主线说明：`Docs/table_pipeline.md`
  - 详细实验记录：`Docs/experiments/`

## 1. 报告目的

本文档用于在 `2026-04-08` 对风速回归方向的全部探索做一次收束，回答四个问题：

1. 从 `src/try/` 的所有相关探索中，哪些路线被验证为有效；
2. 哪些路线只在局部数据域有效，不能直接升级为统一默认主线；
3. 如果必须停止探索，当前应如何给出“最终模型”；
4. 后续如需继续迭代，最应该从哪里接续，而不是重新铺开全矩阵实验。

本文档聚焦“风速回归”主线。以下探索只做简要归档，不作为最终模型选择主证据：

- `001-008`
  - 早期频谱、边界、视频 ROI 与 case5 可视化诊断；
- `018`
  - 结构基频识别；
- `021-023`
  - case5 视频手工标注与视频侧验证。

## 2. 统一实验口径

### 2.1 数据与清洗

- 主数据入口：
  - `data/final/dataset_manifest.csv`
  - `data/final/datasets/工况{ID}.csv`
- `2026-04-05` 起，主线标签与元数据统一由 manifest 管理，不再从文件名解析。
- 当前表格与时序主线统一复用：
  - 删除首尾连续缺失段；
  - 中间连续缺失 `<=5` 行线性插值；
  - 中间连续缺失 `>5` 行删除并切段；
  - 仅在连续段内切窗；
  - 忽略 `WSMS00005.*`。

### 2.2 评估

- 分组单位始终是“工况”，不是“窗口”；
- `final` 主域统一以 `Leave-One-Condition-Out` 作为核心评估；
- `added` 与 `added2` 主要承担外部验证、域偏移诊断与可部署性回放；
- 除特别说明外，风速指标统一优先看 `case_mae`。

### 2.3 统一判断标准

- 若目标是 `final` 域内最优，优先看 `full final LOCO`；
- 若目标是统一主线，优先看：
  - `final`
  - `added`
  - `added2`
  三域是否同时可接受；
- 若目标是可部署默认链，则必须排除依赖 `true_rpm` 的上界写法。

## 3. 探索总览

| 阶段 | try 范围 | 核心问题 | 结论 |
| --- | --- | --- | --- |
| 早期诊断 | `001-008` | 数据、频谱、边界与视频可视化是否存在明显口径问题 | 为主线提供清洗与频谱直觉，但不直接决定最终风速模型 |
| 表格基线 | `009-011` | 手工特征 + 经典模型的最强基线是什么 | `Ridge + VIB_FT_RPM` 成为早期主基线，rpm-free 最优为 `Ridge + VIB_FT` |
| 原始时序 shortlist | `012-017` | 端到端原始时序路线能否超过表格特征 | `TinyTCN` 超过表格基线；`PatchTST / MICN / SAMformer` 在当前口径下失败 |
| TinyTCN 深化 | `013-014, 025-029` | 单窗长、多尺度、注意力、双流等 direct learned 路线谁最强 | `TinyTCN 2s+8s late fusion` 成为 `final` 域内 SOTA |
| RPM 改写 | `019-020, 024, 043` | 先估 `rpm` 再映射 `wind` 是否更稳 | `pred_rpm` 在 `final` 可用，但在 `added` 崩坏，不可作为默认 deployable 支线 |
| added 域诊断 | `030-040` | `added` 失配来自哪里，能否修复 | 主矛盾是应变侧域偏移；可迁移应变信息收敛到 `3.0-6.0Hz` |
| 解析 + learned 融合 | `041-045` | `rpm-first` 与 learned 中频能否形成更强融合 | 对旧 `added 21-24` 成立，最佳稳定写法为 `rpm_knn4 + midband @ w≈0.3`；但不能直接升级为统一主线 |
| 统一修正与 gate | `046-061` | residual / gate / prototype / alignment 能否做统一主干 | 有局部正信号，但没有同时保住 `final` 与外部域 |
| added2 复核 | `062-064` | 旧结论能否迁移到新的外部域 | `rpm-first` 仍最稳；`direct 2s+8s` 和 `midband` 都未自然迁移 |

## 4. 各阶段详细结论

### 4.1 早期诊断与主线口径固定：`001-008`

`001-008` 的作用主要是建立后续探索的共同前提，而不是直接筛模型：

- 频谱与 case5 相关可视化说明，风速与转频信息确实值得进入主线；
- 边界与缺失段检查支持“连续段内切窗”的后续统一口径；
- 视频 ROI 与 case5 中段分析为视频 RPM 支线提供了可复核样本；
- 这些探索没有产生可直接升级为风速回归最终模型的候选，但避免了后续大规模实验建立在错误数据口径上。

### 4.2 表格特征基线阶段：`009-011`

`2026-04-05`，表格主线先完成了可解释基线的筛选：

- `Ridge + VIB_FT_RPM`
  - 早期全局最优；
- `Ridge + VIB_FT`
  - 早期最优 rpm-free 基线。

其意义不在于最终分数，而在于建立了后续所有深度路线必须超过的基线。

到 `2026-04-05`，正式主线 `src/current/` 记录的稳定结论为：

- 若允许 `rpm`：`Ridge + VIB_FT_RPM`
- 若必须 rpm-free：`Ridge + VIB_FT`

这条线为后面所有深度模型提供了“是否值得继续”的最低门槛。

### 4.3 原始时序模型首轮筛选：`012-017`

`012` 先验证了“原始窗口路线是否有希望”：

- `TabularReference_G6_Ridge`
  - `case_mae = 0.4045`
- `MiniRocketLikeRidge`
  - `case_mae = 0.6039`
- `RawFlattenMLP`
  - `case_mae = 0.8440`
- `RawFlattenRidge`
  - `case_mae = 0.9174`

这一步说明：

- 不是所有原始时序路线都会自然超过表格特征；
- 直接展平或轻量随机卷积不足以替代表格基线。

`013` 的轻量 CNN / TCN full 复核给出了第一次明确升级：

- `TinyTCN`
  - `case_mae = 0.3030`
  - `case_rmse = 0.4930`
- `Tiny1DCNN`
  - `case_mae = 0.3340`
  - `case_rmse = 0.5151`
- `TabularReference_G6_Ridge`
  - `case_mae = 0.4045`
  - `case_rmse = 0.7106`

到 `2026-04-05`，风速回归的主候选第一次从表格模型切换为 `TinyTCN`。

`015-017` 又把几条更“大”的外部模型骨干拉进来做 LOCO：

- `PatchTST`
  - `case_mae = 4.4532`
- `MICN`
  - `case_mae = 8.3139`
- `SAMformer`
  - `case_mae = 4.4661`

这说明在当前 `50Hz / 5s / LOCO / 小样本工况级泛化` 口径下，直接复用大时序骨干并不成立。

### 4.4 TinyTCN 深化与 `final` 域内 SOTA：`014, 025-029`

`014` 的窗长扫描显示：

- `TinyTCN @ 5s`
  - `case_mae = 0.3030`
- `TinyTCN @ 4s`
  - `case_mae = 0.3115`
- `TinyTCN @ 8s`
  - `case_mae = 0.3136`
- `TinyTCN @ 2s`
  - `case_mae = 0.3236`

单窗长最优点落在 `5s`。

`025` 的边界误差检查说明，如果继续沿 direct TinyTCN 主线，后续值得注意“边界窗口降权或稳态优先”，但这一步没有推翻 `5s` 主线。

`026` 则是本项目风速 direct learned 路线的关键转折点：

- `TinyTCN@5s`
  - `case_mae = 0.3030`
- `TinyTCN_multiscale_late_fusion_2s_8s`
  - `case_mae = 0.1685`
  - `case_rmse = 0.2522`

这一步说明：

- 短窗与长窗的信息在难工况上明显互补；
- `2s+8s` 晚融合成为 `final` 域内最强结果。

随后 `027-029` 的双流与通道注意力 quickcheck 没有把这条 direct multiscale 路线继续明显推高，因此截至 `2026-04-08`，`final` 域内的 direct learned SOTA 仍是：

- `TinyTCN_multiscale_late_fusion_2s_8s`
  - `case_mae = 0.1685`

### 4.5 RPM 改写与 deployable 支线分化：`019-024, 043`

`019-024` 把问题改写成 `signal -> rpm`，确认：

- TinyTCN 可以稳定做 `rpm` 回归；
- `rpm` 回归的较优窗长落在 `3.0s` 附近。

但 `043` 的关键结论是：

- `pred_rpm -> wind` 在 `final LOCO` 上仍可工作；
- `pred_rpm -> wind` 在 `added` 外部域上整体崩坏。

代表结果：

- `final LOCO`
  - `pred_rpm_3.0s -> ridge_rpm_to_wind`
  - `case_mae = 0.3262`
- `added`
  - `pred_rpm_2.0s -> rpm_linear`
  - `case_mae = 1.8886`

这说明：

- `pred_rpm` 支线可以保留为研究路线；
- 但不能进入统一默认部署链。

### 4.6 added 域诊断与中频应变收敛：`030-040`

`034-037` 系列把 `added 21-24` 的问题定位清楚：

- 不是输入列被裁坏；
- 不是简单的“难工况污染训练池”；
- 主要是应变侧域偏移；
- `工况22` 是最强的异常机制点。

`037` 的模态对照结果尤其关键：

- `rpm_knn4`
  - `case_mae = 0.2293`
- `full_final_pool | acc_only`
  - `case_mae = 0.3553`
- `full_final_pool | all_channels`
  - `case_mae = 3.2158`
- `full_final_pool | strain_only`
  - `case_mae = 2.3854`

这说明 added 失配的主驱动来自应变侧，而不是加速度侧。

`038-040` 进一步收敛到“哪些应变信息可以保留”：

- 应变 `>2Hz high-pass` 能显著缓解 added 崩坏；
- 更稳定的可迁移频带收敛到 `3.0-6.0Hz`。

`040` 的代表结果：

- `rpm_knn4`
  - `case_mae = 0.2293`
- `TinyTCN | strain_bandpass_3.0_6.0Hz`
  - `case_mae = 0.2584`
- `acc_only + strain_bandpass_3.0_6.0Hz`
  - `case_mae = 0.2468`

这说明：

- 原始全频应变不能直接保留；
- 但 `3.0-6.0Hz` 中频应变确实有独立有效信息。

### 4.7 解析主干与 learned 中频增强：`041-045`

`041-042` 证明旧 `added 21-24` 上存在一条稳定增强线：

- `rpm_knn4`
  - `case_mae = 0.2293`
- `rpm_knn4 + TinyTCN all_channels midband @ w=0.3`
  - `case_mae mean = 0.1627`
  - `case_mae std = 0.0223`

并且 `w=0.3` 比 `w=0.5` 更稳。

`043_1-043_3` 又把“有真实 `rpm` 的上界写法”改写成可部署版本：

- `fft_peak_1x_whole -> rpm_knn4`
  - `added case_mae = 0.1860`
- `fft_peak_1x_whole -> rpm_knn4 + TinyTCN all_channels midband @ w=0.3`
  - `added case_mae mean = 0.1675`

这一步的真正意义是：

- 旧 `added` 上，`rpm-first` 与 learned midband 不是替代关系，而是互补关系；
- FFT 已经可以替代 `true_rpm` 进入部署链；
- 但这条增强线当时只覆盖 `added 21-24`，并未自动成为 `final + added` 的统一主线。

`045` 把 `added` 并入训练池后的统一 `LOOCV` quickcheck 又进一步说明：

- direct `TinyTCN 2s/8s` 在统一池上明显退化；
- 旧 `final` 域内成立的 `2s+8s` 优势不能靠简单并池保住。

### 4.8 统一修正、gate 与表征方向：`046-061`

`046` 先证明：如果有 `true_rpm`，风速问题确实可以被优先重写成 `rpm -> wind`，而“再叠一个 acc residual”并没有形成足够强的新主线。

`047-050` 的 gate 系列结论是：

- 两专家之间确实存在很大可门控空间；
- 表格 gate 不够强；
- `TCN gate` 在 added 上已经很强；
- 但所有 gate 版本都没有真正把 `final` 退化压回 `base_only`。

代表结果：

- `base_only(final SOTA = TinyTCN 2s+8s)`
  - `final case_mae = 0.1685`
- `TCN soft gate`
  - `all_labeled case_mae = 0.2301`
  - `final case_mae = 0.2567`
  - `added case_mae = 0.1036`
- `TCN conservative gate`
  - `all_labeled case_mae = 0.2300`
  - `final case_mae = 0.2453`
  - `added case_mae = 0.1571`

因此到 `2026-04-08`，gate 方向更像“强增强候选”，还不是可直接替代主线的最终解。

`051-059` 的 embedding residual / support-window / delta-only trigger 方向说明：

- `TinyTCN embedding` 确实携带 added 局部修正信号；
- 但这类修正一旦全局开启，仍会伤害 `final`；
- 最强角色仍是“可选局部修正器”，而不是统一主预测器。

`060-061` 的 prototype alignment 系列则确认：

- 代表性 hard case 上出现过独立正信号；
- 但放回 `full final LOCO` 和 `added` 同口径后，没有超过现有 SOTA。

截至 `2026-04-08`，这些方向都只能保留为后续研究储备，不能写成当前最终模型。

### 4.9 added2 复核与统一主线回退：`062-064`

`062` 先把 `added2 25-30` 的结构拆清楚：

- `25-27`
  - 更像高转速高风速新支路；
- `28-29`
  - 更像 `final` 主簇延伸；
- `30`
  - 再次表现出 `工况22` 式低频应变异常。

`063` 回放 `final` 训练的 direct `TinyTCN 2s+8s`：

- `added2 | TinyTCN@2s`
  - `case_mae = 2.1985`
- `added2 | TinyTCN 2s+8s late fusion`
  - `case_mae = 2.2897`

这说明 `final` 域内最强的 direct learned 路线，没有自然迁移到 `added2`。

`064` 又复核了旧 `added` SOTA 到 `added2` 的迁移：

- `added2 | rpm_knn4`
  - `case_mae mean = 0.8131`
- `added2 | rpm_knn4 + TinyTCN midband @ w=0.3`
  - `case_mae mean = 1.0126`
- `added2 | direct TinyTCN 2s+8s`
  - `case_mae = 2.2897`

这一步非常关键，因为它把此前所有“局部 SOTA”重新排序成了统一工程结论：

- `added2` 上更稳的默认候选是 `rpm_knn4`；
- 旧 `added` 上成立的 `midband` 增强，没有迁移到 `added2`；
- `final` 上成立的 direct `2s+8s`，也没有迁移到 `added2`。

## 5. 最终候选矩阵

| 目标 | 最优或主推候选 | 指标 | 当前结论 |
| --- | --- | --- | --- |
| `final` 域内最高精度 | `TinyTCN_multiscale_late_fusion_2s_8s` | `case_mae = 0.1685` | 作为 `final LOCO` 的 direct learned SOTA 保留 |
| 旧 `added 21-24` 上界增强 | `rpm_knn4 + TinyTCN midband @ w=0.3` | `case_mae mean = 0.1627` | 只对旧 `added` 有稳定证据，不可直接统一 |
| 旧 `added 21-24` 可部署增强 | `fft_peak_1x_whole -> rpm_knn4 + TinyTCN midband @ w=0.3` | `case_mae mean = 0.1675` | 作为 added deployable 增强线保留 |
| 旧 `added 21-24` 可部署保守基线 | `fft_peak_1x_whole -> rpm_knn4` | `case_mae = 0.1860` | 强可部署解析基线 |
| `added2` 更稳默认主干 | `rpm_knn4` | `case_mae mean = 0.8131` | 截至 `2026-04-08` 最稳统一主干 |
| 经典可解释保底基线 | `Ridge + VIB_FT_RPM` | 早期主线最优 | 保留为正式表格参考，不再作为主候选 |

## 6. 最终模型结论

### 6.1 如果必须只给一个“统一最终模型”

截至 `2026-04-08`，若要求一个同时考虑 `final / added / added2` 的统一稳健主干，当前更合理的最终模型是：

- `rpm_knn4`

原因不是它在每个子域都是绝对最优，而是：

- `final` 域内虽然不是 direct learned SOTA，但仍在可接受量级；
- `added` 上显著稳于 direct learned 外推；
- `added2` 上也优于 `midband` 增强与 direct `2s+8s`；
- 后续所有统一 residual / gate / prototype 方案都还没有稳定超过它。

### 6.2 如果必须给一个“当前可部署最终模型”

若真实 `rpm` 不可直接使用，当前更合理的 deployable 默认链是：

- `fft_peak_1x_whole -> rpm_knn4`

原因：

- `pred_rpm TinyTCN -> wind` 在 `added` 已明确崩坏；
- FFT 解析链在 `added` 已显著优于 `pred_rpm` 支线；
- 这条链的行为也比 direct learned 外推更容易解释与诊断。

### 6.3 如果汇报目标是“主数据集上的最高精度模型”

若报告需要突出 `final` 主数据集上的最佳结果，可以把以下模型写成“`final` 域内最优模型”：

- `TinyTCN_multiscale_late_fusion_2s_8s`

但必须同步注明：

- 该结论只对 `final LOCO` 成立；
- 截至 `2026-04-08`，它不能直接外推到 `added` 与 `added2`。

### 6.4 如果允许保留一个“增强版候选”

仅在需要单独讨论旧 `added 21-24` 外部域时，可保留：

- `fft_peak_1x_whole -> rpm_knn4 + TinyTCN all_channels midband(3.0-6.0Hz) @ w≈0.3`

但必须明确写出适用边界：

- 该结论在 `2026-04-07` 对旧 `added 21-24` 成立；
- 该结论在 `2026-04-08` 没有迁移到 `added2`；
- 因此它不是统一最终主线，只是“旧 added 的增强版可部署候选”。

## 7. 被淘汰路线与淘汰原因

### 7.1 表格模型没有继续做主线

原因：

- 被 `TinyTCN` 明显超过；
- 但保留为解释性基线和回退参考。

### 7.2 外部大模型骨干没有继续推进

包括：

- `PatchTST`
- `MICN`
- `SAMformer`

原因：

- 在当前 LOCO 小样本工况级泛化口径下，表现明显劣于轻量 `TinyTCN`；
- 不值得继续扩大适配成本。

### 7.3 `pred_rpm -> wind` 没有进入默认候选矩阵

原因：

- `final` 域内可用；
- `added` 域外高估崩坏；
- 当前最大问题在 `pred_rpm` 的外部泛化，不在 `rpm -> wind` 映射形式。

### 7.4 全频应变路线没有保留

原因：

- added 外部高估主由应变侧低频偏移驱动；
- 唯一保留下来的应变信息，是 `3.0-6.0Hz` 中频带。

### 7.5 unified gate / residual / prototype 没有升级为最终模型

原因：

- 都有局部正信号；
- 但都没有同时保住 `final` 与外部域；
- 截至 `2026-04-08` 都还处于“研究型增强器”阶段。

## 8. 建议的报告写法

### 8.1 如果面向课程答辩

建议把结论拆成两句：

1. `2026-04-08` 在 `final` 域内，最佳风速回归模型为 `TinyTCN_multiscale_late_fusion_2s_8s`，`case_mae = 0.1685`。
2. `2026-04-08` 若考虑外部工况与可部署性，统一更稳的默认主干应优先参考 `rpm_knn4`，无真实 `rpm` 时可部署链优先参考 `fft_peak_1x_whole -> rpm_knn4`。

这样可以同时保留：

- 学术层面的最好结果；
- 工程层面的更稳结论。

### 8.2 如果面向工程交接

建议直接把主结论写成：

- 默认主干：`rpm_knn4`
- 默认 deployable 链：`fft_peak_1x_whole -> rpm_knn4`
- `TinyTCN 2s+8s` 作为 `final` 域内高精度分支保留；
- `midband` 与 gate 家族作为后续增强方向保留，但默认关闭。

## 9. 后续最小增量方向

如果 `2026-04-08` 之后允许继续做极少量增量实验，更值得继续的是：

1. 围绕 `rpm-first` 主干做 route / trust gate，而不是重新追统一 direct learned 主线；
2. 把 `added2` 的 `25-27`、`28-29`、`30` 分成不同子域分别处理，而不是整块并池；
3. 继续把 `TinyTCN embedding` 用作“局部修正信号”，但必须带显式 final 保护；
4. 优先做“何时禁止增强”的保护型 gate，而不是继续放大增强强度。

## 10. 一句话总结

截至 `2026-04-08`，本项目的探索已经形成清晰分层结论：

- `TinyTCN 2s+8s` 是 `final` 域内最强的 direct wind 模型；
- `rpm_knn4` 是跨 `final / added / added2` 更稳的统一主干；
- `fft_peak_1x_whole -> rpm_knn4` 是当前更合理的 deployable 默认链；
- `midband`、gate、prototype、alignment 都证明了局部增强是可能的，但在 `2026-04-08` 还不足以替代默认主线。
