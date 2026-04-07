# 解析基线与 Learned 中频分支复核（2026-04-07）

- 状态：`historical`
- 首次确认：`2026-04-07`
- 最近复核：`2026-04-07`
- 替代关系：
  - 关于固定权重默认值的当前结论，已由 `Docs/experiments/rpm_learned_midband_multiseed_stability_check_2026-04-07.md` 取代；
  - 本文档保留为单次 seed 发现记录。
- 数据范围：
  - 主训练域：`data/final/` 的带标签工况
  - added 外部工况：`data/added/` 的 `工况21-24`
  - 重点工况：`工况22`
- 代码口径：
  - `src/try/041_rpm_vs_learned_midband_check/`
- 证据入口：
  - `outputs/try/041_rpm_vs_learned_midband_check/variant_config_table.csv`
  - `outputs/try/041_rpm_vs_learned_midband_check/case_level_predictions.csv`
  - `outputs/try/041_rpm_vs_learned_midband_check/summary.csv`
  - `outputs/try/041_rpm_vs_learned_midband_check/case22_focus.csv`
  - `outputs/try/041_rpm_vs_learned_midband_check/decision_reference.csv`
  - `outputs/try/041_rpm_vs_learned_midband_check/summary.md`

## 1. 目标

在 `040` 已经把应变 learned 分支收敛到 `3.0-6.0Hz` 后，进一步验证：

- `rpm_knn4` 解析基线与 learned 中频分支谁更稳；
- 轻量线性模型是否足以利用这段中频应变；
- `rpm + learned` 混合是否已经优于两者单独使用。

## 2. 方法口径

- 训练池：
  - `full_final_pool`
- 评估集：
  - `data/added/` 的 `工况21-24`
- 方案矩阵：
  - `rpm_knn4`
  - `TinyTCN | strain_only | 3.0-6.0Hz`
  - `TinyTCN | all_channels_midband | 3.0-6.0Hz`
  - `Ridge | strain_only | 3.0-6.0Hz`
  - `rpm_knn4 + TinyTCN all_channels midband`
    - learned 权重：`0.3 / 0.5 / 0.7`

## 3. 当前结果

### 3.1 [2026-04-07] 单次 seed 下 `rpm_knn4 + learned midband` 已优于纯解析和纯 learned

- 状态：`historical`
- 首次确认：`2026-04-07`
- 最近复核：`2026-04-07`

当前最优结果：

- `fusion_rpm_knn4__tinytcn_all_channels_midband__w0.5`
  - `case_mae = 0.1372`
  - `case_rmse = 0.1848`
  - `mean_signed_error = -0.0333`
  - `case22_abs_error = 0.0073`

对照：

- `rpm_knn4`
  - `case_mae = 0.2293`
- `TinyTCN all_channels midband`
  - `case_mae = 0.2926`

这说明：

- 在这次单次 seed 复核中，added 方向的最优候选已经不再是纯解析或纯 learned；
- 该次单次 seed 的最优方案为：
  - `rpm_knn4 + TinyTCN all_channels midband`
  - 且 learned 权重约 `0.5`

### 3.2 [2026-04-07] `3.0-6.0Hz` 中频应变里确实存在需要 learned 模型才能利用的结构信息

- 状态：`historical`
- 首次确认：`2026-04-07`
- 最近复核：`2026-04-07`

关键对照：

- `TinyTCN strain_only midband`
  - `case_mae = 0.2935`
- `Ridge strain_only midband`
  - `case_mae = 0.9476`

这说明：

- `3.0-6.0Hz` 这段应变信息不是“简单线性关系就能吃干净”的信号；
- 当前这段中频应变里确实存在非线性 / 时序结构；
- 因此如果保留这条应变支线，更合理的做法是保留 learned 模型，而不是退回轻量线性回归。

### 3.3 [2026-04-07] learned 分支更适合保留加速度上下文，而不是只看应变

- 状态：`historical`
- 首次确认：`2026-04-07`
- 最近复核：`2026-04-07`

关键对照：

- `TinyTCN all_channels midband`
  - `case_mae = 0.2926`
  - `case22_abs_error = 0.2805`
- `TinyTCN strain_only midband`
  - `case_mae = 0.2935`
  - `case22_abs_error = 0.3447`

这说明：

- 两者整体误差接近，但带加速度上下文的 learned 分支在 `工况22` 上更稳；
- 当前更合理的 learned 中频支线应写成：
  - `all_channels + strain(3.0-6.0Hz)`
  - 而不是完全去掉加速度

### 3.4 [2026-04-07] 单次 seed 下 `工况22` 已被混合方案几乎完全修平

- 状态：`historical`
- 首次确认：`2026-04-07`
- 最近复核：`2026-04-07`

`工况22` 的 case 级结果：

- `rpm_knn4 + TinyTCN all_channels midband @ w=0.5`
  - `pred = 3.4073`
  - `abs_error = 0.0073`
- `rpm_knn4 + TinyTCN all_channels midband @ w=0.3`
  - `abs_error = 0.1019`
- `rpm_knn4 + TinyTCN all_channels midband @ w=0.7`
  - `abs_error = 0.1166`
- `rpm_knn4`
  - `abs_error = 0.2658`
- `TinyTCN all_channels midband`
  - `abs_error = 0.2805`

这说明：

- `工况22` 这个 added 方向最难点，当前已经可以被“解析 + learned 中频分支”混合方案几乎完全修平；
- 这进一步支持：
  - 不应把 `rpm_knn4` 与 learned 中频分支视作替代关系
  - 更应视作互补关系

## 4. 当前判断

`2026-04-07` 的这轮单次 seed 复核支持以下历史判断：

- 当时的 added 方向最优候选写作：
  - `rpm_knn4 + TinyTCN all_channels midband`
  - learned 权重约 `0.5`
- `rpm_knn4` 仍应保留为强解析基线；
- `3.0-6.0Hz` 中频应变 learned 分支已经证明有独立价值；
- 该结论后续已按当时计划进入重复种子 / 稳定性复核；
- 关于默认固定权重的当前结论，请以 `042` 的多随机种子复核文档为准。
