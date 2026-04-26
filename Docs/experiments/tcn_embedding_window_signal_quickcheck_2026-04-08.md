# TinyTCN embedding 窗长信号 quickcheck（2026-04-08）

- 状态：`current`
- 首次确认：`2026-04-08`
- 最近复核：`2026-04-08`
- 数据范围：
  - `final` 代表性 holdout：`工况1 / 3 / 17 / 18`
  - `added`：`工况21-24`
- 代码口径：
  - `src/try/052_tcn_embedding_window_signal_quickcheck/`
- 证据入口：
  - `outputs/try/052_tcn_embedding_window_signal_quickcheck/case_level_predictions.csv`
  - `outputs/try/052_tcn_embedding_window_signal_quickcheck/summary_by_domain.csv`
  - `outputs/try/052_tcn_embedding_window_signal_quickcheck/nearest_neighbors.csv`
  - `outputs/try/052_tcn_embedding_window_signal_quickcheck/summary.md`

## 1. 目标

在 `051` 已经发现 `2s embedding residual` 对 added 有正信号、但还不清楚该不该换成 `5s` 或 `2s+8s` 的前提下，做一个更小成本的代表性 holdout quickcheck：

- 不做全量 `LOOCV`
- 只比较最关心的时序窗长：
  - `2s`
  - `5s`
  - `8s`
  - `2s+8s`

问题是：

- 对 `rpm_knn4` 主干来说，哪类 embedding residual 更像“可保留的修正信号”。

## 2. 方法口径

- holdout 工况：
  - `final`：`1 / 3 / 17 / 18`
  - `added`：`21 / 22 / 23 / 24`
- 训练池：
  - 每次 holdout 时用其余全部带标签工况训练
- 时序模型：
  - `TinyTCN`
- 变体：
  - `rpm_knn4`
  - `TinyTCN direct @ 2s / 5s / 8s`
  - `TinyTCN direct 2s+8s late fusion`
  - `rpm_knn4 + embedding_residual_knn4_{2s,5s,8s} @ w=0.5`
  - `rpm_knn4 + embedding_residual_knn4_avg_2s_8s @ w=0.5`
  - `rpm_knn4 + embedding_residual_knn4_concat_2s_8s @ w=0.5`

其中：

- `avg_2s_8s`
  - 先分别算 `2s` 与 `8s` residual 修正，再做平均
- `concat_2s_8s`
  - 把 `2s` 与 `8s` 的 case embedding 直接拼接，再做同一个 kNN residual

## 3. 当前结果

### 3.1 [2026-04-08] 在这组代表性 holdout 上，`2s+8s concat residual` 是当前最强的 embedding correction 候选

`focus_all` 汇总：

- `rpm_knn4`
  - `case_mae = 0.6444`
- `rpm_knn4 + residual_2s @ w=0.5`
  - `case_mae = 0.6028`
- `rpm_knn4 + residual_5s @ w=0.5`
  - `case_mae = 0.6070`
- `rpm_knn4 + residual_8s @ w=0.5`
  - `case_mae = 0.6349`
- `rpm_knn4 + residual_avg_2s_8s @ w=0.5`
  - `case_mae = 0.6164`
- `rpm_knn4 + residual_concat_2s_8s @ w=0.5`
  - `case_mae = 0.6006`

这说明：

- 当前小规模证据支持“多尺度 embedding residual”优于单一 `2s / 5s / 8s`；
- 其中更强的形式不是后平均，而是直接拼接 `2s + 8s` 表征；
- 这说明短窗和长窗在 residual 方向上更像互补，而不是完全替代。

### 3.2 [2026-04-08] `5s residual` 没有明显超过 `2s residual`

`focus_all` 对照：

- `residual_2s @ w=0.5`
  - `case_mae = 0.6028`
- `residual_5s @ w=0.5`
  - `case_mae = 0.6070`

这说明：

- 把 `051` 的 `2s` 直接改成 `5s`，当前没有带来更强信号；
- `5s` 在 direct 风速回归里更强，不代表它在 residual embedding 里也更强；
- 当前 residual 这条线仍更像偏向短窗或多尺度，而不是单独长窗。

### 3.3 [2026-04-08] `2s+8s concat residual` 同时改善了 `final_focus` 与 `added_focus`

- `final_focus`
  - `rpm_knn4 = 1.0595`
  - `residual_concat_2s_8s = 1.0159`
- `added_focus`
  - `rpm_knn4 = 0.2293`
  - `residual_concat_2s_8s = 0.1852`

这说明：

- 与 `051` 的全量 quickcheck 不同，这组代表性 holdout 上的 residual 信号并不只偏 added；
- 在 `工况3 / 17` 这类 final hard cases 上，residual 也给出了可见改善；
- 不过这个结论当前只覆盖代表性小样本，不应直接外推成全局默认路线。

### 3.4 [2026-04-08] direct TinyTCN 方向在这组 holdout 上仍然明显不如 `rpm-first`

`focus_all` 对照：

- `rpm_knn4`
  - `case_mae = 0.6444`
- `tinytcn_direct_2s`
  - `case_mae = 1.0721`
- `tinytcn_direct_5s`
  - `case_mae = 1.1895`
- `tinytcn_direct_2s_8s_fusion`
  - `case_mae = 1.1409`

这说明：

- 即使换成 `5s` 或 `2s+8s`，direct learned predictor 这条线在双域代表性 holdout 上依然明显弱于 `rpm-first + correction`；
- 当前值得继续追的不是 direct TCN 多尺度，而是多尺度 residual。

## 4. 当前判断

截至 `2026-04-08`，这轮小规模 quickcheck 更支持下面的表达：

- `2s` 不是唯一值得保留的 residual 窗长；
- 但 `5s` 也没有单独证明自己比 `2s` 更强；
- 当前更值得继续推进的候选是：
  - `rpm_knn4 + embedding_residual_concat_2s_8s @ conservative weight`

也就是说：

- 如果下一步继续做 embedding residual，不应只盯单窗长；
- 更合理的是把它重写成：
  - `rpm-first`
  - `multiscale embedding correction`
  - 再加是否开启修正的门控。

## 5. 一句话版结论

截至 `2026-04-08`，在 `工况1 / 3 / 17 / 18 / 21 / 22 / 23 / 24` 这组代表性 holdout 上，`2s+8s` 拼接 embedding 的 residual 修正已经给出比单独 `2s`、单独 `5s` 更强的信号；因此后续若继续追 `TCN embedding` 这条线，更值得把它组织成“多尺度 residual”，而不是只换成 `5s` 单窗长。
