# soft gate quickcheck（2026-04-08）

- 状态：`current`
- 首次确认：`2026-04-08`
- 最近复核：`2026-04-08`
- 数据范围：
  - `final` 带标签工况 `1, 3-20`
  - `added` 带标签工况 `21-24`
- 代码口径：
  - `src/try/047_soft_gate_quickcheck/`
- 证据入口：
  - `outputs/try/047_soft_gate_quickcheck/gate_feature_table.csv`
  - `outputs/try/047_soft_gate_quickcheck/case_level_predictions.csv`
  - `outputs/try/047_soft_gate_quickcheck/summary_by_variant.csv`
  - `outputs/try/047_soft_gate_quickcheck/summary_by_domain.csv`
  - `outputs/try/047_soft_gate_quickcheck/summary.md`

## 1. 目标

在不使用原始波形的前提下，验证“两个专家 + soft gate 比例输出”是否有正信号。

本轮不先做硬二分路由，而是直接学习：

- `g in [0, 1]`

最终预测写成：

- `pred = (1 - g) * pred_base + g * pred_enhanced`

## 2. 专家定义

### 2.1 `base expert`

- `TinyTCN_multiscale_late_fusion_2s_8s`

口径：

- `final`：复用 `026` 的 `full19` case-level `LOCO`
- `added`：复用 `035` 的 `full_final_pool` 外部预测

### 2.2 `enhanced expert`

- `true_rpm -> rpm_knn4 + TinyTCN all_channels midband(3.0-6.0Hz) @ w=0.3`

口径：

- `final`：复用 `044` 中 `fusion_true_rpm_knn4__tinytcn_all_channels_midband_final_loco__w0.3`
  - 当前用 `3` 个已落盘 seed 的 case 级预测均值
- `added`：复用 `042` 中 `fusion_rpm_knn4__tinytcn_all_channels_midband__w0.3`
  - 当前用多 seed case 级预测均值

## 3. gate 特征与模型

### 3.1 特征工程

gate 只使用 `case-level` 可解释特征：

- `true_rpm`
- `pred_base`
- `pred_enhanced`
- `pred_gap`
- `abs_pred_gap`
- `hour_sin`
- `hour_cos`
- `strain_low_ratio_median`
- `strain_mid_ratio_median`
- `strain_low_over_mid`
- `strain_rms_median`
- `acc_energy_median`
- `acc_peak_freq_median`
- `strain_acc_rms_ratio`
- `missing_ratio_in_common_cols`
- `edge_removed_ratio`

### 3.2 监督信号

对每个工况构造最优连续比例：

- `g* = clip((y - pred_base) / (pred_enhanced - pred_base), 0, 1)`

这意味着 gate 学习的不是“样本来自哪个目录”，而是：

- 在当前样本上，增强路线该开多大

### 3.3 比较模型

- `base_only`
- `enhanced_only`
- `global_weight_cv`
  - 每个 fold 用训练工况搜索单一全局比例
- `ridge_gate`
- `hgb_gate`
- `oracle_soft_gate`
  - 仅作为上界参考

## 4. 当前结果

### 4.1 [2026-04-08] 两个专家之间存在很大的“可门控空间”

`oracle_soft_gate` 结果：

- `final`
  - `case_mae = 0.0963`
- `added`
  - `case_mae = 0.0238`
- `all_labeled`
  - `case_mae = 0.0837`

对照：

- `base_only`
  - `final = 0.1685`
  - `added = 2.8047`
- `enhanced_only`
  - `final = 0.4281`
  - `added = 0.1571`

这说明：

- 当前不是“两个专家差不多，gate 没必要”；
- 而是两个专家之间确实存在很大的互补空间；
- 问题不在“要不要 gate”，而在“第一版 gate 还没学好”。

### 4.2 [2026-04-08] 第一版 `hgb_gate` 已明显优于固定走单个专家

`all_labeled` 对照：

- `base_only`
  - `case_mae = 0.6270`
- `enhanced_only`
  - `case_mae = 0.3809`
- `global_weight_cv`
  - `case_mae = 0.3863`
- `hgb_gate`
  - `case_mae = 0.3219`

这说明：

- 当前 `HistGradientBoostingRegressor` 已经不只是学到“接近某个固定全局比例”；
- 它已经比 `global_weight_cv` 更好，说明它确实在学习条件化权重。

### 4.3 [2026-04-08] 第一版 `hgb_gate` 还没有压住 `final` 退化

`final` 对照：

- `base_only`
  - `case_mae = 0.1685`
- `hgb_gate`
  - `case_mae = 0.2366`

这说明：

- 当前第一版 soft gate 还不能升级成最终默认路线；
- 问题不在“soft gate 思路错误”，而在：
  - 当前样本规模太小；
  - 4 个 added 样本不足以把边界学稳；
  - 第一版特征和模型还不够保守。

### 4.4 [2026-04-08] `ridge_gate` 当前不适合作为默认 gate

- `ridge_gate`
  - `final = 0.3015`
  - `added = 1.6445`
  - `all_labeled = 0.5351`

这说明：

- 当前 route / gate 关系并不是简单线性结构；
- 仅靠线性回归 `g` 很容易学成“所有样本都给中间权重”，效果明显不够。

## 5. 当前判断

截至 `2026-04-08`，这轮 quickcheck 支持下面的判断：

- `soft gate` 的问题定义是对的；
- “输出比例而不是先做硬二分”是值得继续追的方向；
- 当前 `gate` 已经比“固定走某一个专家”更好，但还没达到可直接升级默认路线的程度；
- 当前第一优先级不是换回原始波形或继续扩大 TCN，而是：
  - 让 gate 更保守；
  - 更强调“不要伤害 final”；
  - 再继续提升 added 保留率。

更具体地说，下一步更值得验证的是：

- 让 gate 先学：
  - `enhanced_better` 二分类
  - 或 `是否允许开启增强`
- 再把连续比例限制在：
  - `0 / 0.3 / 0.5 / 1.0`
  - 而不是一开始就自由回归整个 `[0, 1]`
- 同时继续强化与 added 相关的几个关键特征：
  - 夜间时间特征
  - 应变低频占比
  - `pred_base - pred_enhanced`

## 6. 一句话版结论

截至 `2026-04-08`，soft gate 已经证明“按样本输出比例”这条路是值得继续的，但当前第一版 `hgb_gate` 还没有压住 `final` 退化，因此更合理的下一步不是放弃 gate，而是把 gate 做得更保守、更接近“先判定能不能开增强，再决定开多大”。
