# conservative gate quickcheck（2026-04-08）

- 状态：`current`
- 首次确认：`2026-04-08`
- 最近复核：`2026-04-08`
- 数据范围：
  - `final` 带标签工况 `1, 3-20`
  - `added` 带标签工况 `21-24`
- 代码口径：
  - `src/try/048_conservative_gate_quickcheck/`
- 证据入口：
  - `outputs/try/048_conservative_gate_quickcheck/dataset_table.csv`
  - `outputs/try/048_conservative_gate_quickcheck/case_level_predictions.csv`
  - `outputs/try/048_conservative_gate_quickcheck/summary_by_variant.csv`
  - `outputs/try/048_conservative_gate_quickcheck/summary_by_domain.csv`
  - `outputs/try/048_conservative_gate_quickcheck/summary.md`

## 1. 目标

在 `047` 已证明 soft gate 的问题定义成立、但第一版连续回归 gate 仍明显伤害 `final` 的前提下，进一步验证：

- 更保守的 gate 设计是否更适合当前小样本；
- 是否应该优先学“开不开增强”，而不是直接回归任意连续比例；
- 离散档位 gate 是否比连续回归更稳。

## 2. 方法口径

### 2.1 专家定义

- `base expert`
  - `TinyTCN_multiscale_late_fusion_2s_8s`
- `enhanced expert`
  - `true_rpm -> rpm_knn4 + TinyTCN all_channels midband(3.0-6.0Hz) @ w=0.3`

### 2.2 gate 特征

继续复用 `047` 的 `case-level` 特征：

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

### 2.3 保守 gate 变体

#### 二分类 gate

- 目标：
  - 只预测 `enhanced_better`
- 变体：
  - `binary_logit_hard`
  - `binary_hgb_hard`
  - 以及更保守的 `threshold = 0.65` 版本

#### 离散档位 gate

把 oracle 比例映射到最近档位：

- `0.0 / 0.3 / 0.5 / 1.0`

变体：

- `bucket_logit`
- `bucket_hgb`

#### 两阶段 gate

- 第 1 阶段：
  - 先判定是否允许开启增强
- 第 2 阶段：
  - 若允许，再在 `{0.3, 0.5, 1.0}` 中选

变体：

- `two_stage_logit`
- `two_stage_hgb`
- 以及更保守的 `threshold = 0.65` 版本

### 2.4 对照组

- `base_only`
- `enhanced_only`
- `global_weight_cv`
- `047 hgb_gate`
- `oracle_soft_gate`

## 3. 当前结果

### 3.1 [2026-04-08] 保守 gate 没有超过 `047 hgb_gate`

`all_labeled` 关键对照：

- `047 hgb_gate`
  - `case_mae = 0.3219`
- `048 two_stage_hgb`
  - `case_mae = 0.2931`
- `048 bucket_hgb`
  - `case_mae = 0.3436`
- `048 binary_hgb_hard`
  - `case_mae = 0.3607`

这说明：

- 在 `048` 这轮里，最好的保守 gate 是 `two_stage_hgb`；
- 它相对 `047 hgb_gate` 有小幅改善；
- 但提升幅度仍不够大，且更重要的是还没有压住 `final` 退化。

### 3.2 [2026-04-08] 保守 gate 仍然没能压回 `base_only`

`final` 关键对照：

- `base_only`
  - `case_mae = 0.1685`
- `047 hgb_gate`
  - `case_mae = 0.2366`
- `048 two_stage_hgb`
  - `case_mae = 0.2552`
- `048 bucket_hgb`
  - `case_mae = 0.2502`

这说明：

- 更保守的离散 gate 并没有把 `final` 拉回 `base_only` 附近；
- 当前主要问题已经不是“continuous vs discrete”，而是：
  - 样本量太少；
  - added 的 `4` 个工况不足以稳定学出决策边界；
  - 当前 gate 仍然会对一些 `final` 工况过早开启增强。

### 3.3 [2026-04-08] added 上最稳的保守基线仍然是“几乎总是走增强”

`added` 关键对照：

- `enhanced_only`
  - `case_mae = 0.1571`
- `global_weight_cv`
  - `case_mae = 0.2702`
- `048 two_stage_hgb`
  - `case_mae = 0.4733`
- `048 bucket_hgb`
  - `case_mae = 0.7870`

这说明：

- 在 added 这 `4` 个样本上，当前最合理的策略其实接近：
  - “几乎全部走增强”
- 因此 `048` 的保守性主要体现在牺牲 added 收益，并没有换来足够的 `final` 保护。

### 3.4 [2026-04-08] `global_weight_cv` 仍然是一个值得保留的强朴素参考

结果：

- `all_labeled`
  - `case_mae = 0.3863`
- `added`
  - `case_mae = 0.2702`
- `final`
  - `case_mae = 0.4107`

它不优秀，但说明：

- “几乎固定走增强”在当前样本构成下已经能打败很多坏 gate；
- 这进一步说明当前 gate 的难点不在比例形式，而在识别极少数确实不该增强的 `final` 工况。

## 4. 当前判断

截至 `2026-04-08`，`048` 支持下面的判断：

- soft / conservative gate 这条线仍然值得继续；
- 但当前最难的问题不是“要不要离散化”；
- 而是要先识别那些“绝对不能开增强”的 `final` 工况；
- 当前 added 样本太少，使 gate 更容易学成：
  - “大多数时候开增强”
  - 而不是“只在真正需要时开增强”。

因此，下一步更值得做的不是继续扩大 gate 模型，而是先把问题再简化一层：

- 把 gate 改写成：
  - `blocklist / protect-final classifier`
  - 或“是否禁止增强”
- 也就是先学：
  - 哪些样本必须坚持 `base`
- 再在剩余样本中考虑连续比例或离散比例。

## 5. 一句话版结论

截至 `2026-04-08`，保守 gate 比连续回归 gate 更符合当前工程直觉，但它仍没有真正解决“压住 `final` 退化”这个核心问题；下一步更值得优先验证的不是更复杂的比例学习，而是先识别并保护那些不该开启增强的 `final` 工况。
