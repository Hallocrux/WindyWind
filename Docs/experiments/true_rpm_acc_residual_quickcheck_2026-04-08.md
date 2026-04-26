# true rpm 主干 + acc residual quickcheck（2026-04-08）

- 状态：`current`
- 首次确认：`2026-04-08`
- 最近复核：`2026-04-08`
- 数据范围：
  - `final LOCO`：`data/final/` 的带标签工况 `1, 3-20`
  - `added external`：`data/added/` 的 `工况21-24`
- 代码口径：
  - `src/try/046_true_rpm_acc_residual_quickcheck/`
- 证据入口：
  - `outputs/try/046_true_rpm_acc_residual_quickcheck/case_level_predictions.csv`
  - `outputs/try/046_true_rpm_acc_residual_quickcheck/summary_by_variant.csv`
  - `outputs/try/046_true_rpm_acc_residual_quickcheck/summary_by_protocol.csv`
  - `outputs/try/046_true_rpm_acc_residual_quickcheck/summary.md`

## 1. 目标

在 `2026-04-08` 新前提“推理时可直接使用 `true_rpm`”下，快速验证两件事：

- 如果把问题改写成 `rpm -> wind`，当前最强的 `rpm-only` 映射是哪条；
- 在强制让 `rpm` 做主干、加速度只做小修正时，`acc-only residual` 是否还能带来稳定正信号。

## 2. 方法口径

### 2.1 `rpm-only` 对照

统一比较以下工况级映射：

- `linear`
- `ridge`
- `knn4`
- `spline_gam`

### 2.2 `acc residual` 设计

残差建模统一写成：

- `residual = true_wind - rpm_only_pred`

输入只使用窗口级加速度统计特征与 `base_pred`：

- `acc_rms_mean`
- `acc_std_mean`
- `acc_ptp_mean`
- `acc_peak_freq_mean`
- `acc_log_energy_mean`
- `acc_band_ratio_0_2hz_mean`
- `acc_band_ratio_2_5hz_mean`
- `acc_band_ratio_5_10hz_mean`
- `acc_mid_over_low_ratio`
- `acc_high_over_mid_ratio`
- `raw_missing_ratio`
- `rpm`
- `base_pred`

残差模型固定为：

- `StandardScaler + RidgeCV`

同时比较两种残差强度：

- `w = 1.0`
- `w = 0.4`

最终预测写成：

- `pred = rpm_only_pred + w * residual_pred`

其中 `w = 0.4` 的目的不是追求更复杂的融合，而是显式限制加速度只能做小修正，避免它抢走 `rpm` 主导地位。

## 3. 当前结果

### 3.1 [2026-04-08] `true_rpm` 可用时，`rpm-only` 已经是很强的主干

- `final LOCO` 最优 `rpm-only`：
  - `rpm_only__ridge`
  - `case_mae = 0.4053`
- `added external` 最优 `rpm-only`：
  - `rpm_only__knn4`
  - `case_mae = 0.2293`

这说明：

- 在 `true_rpm` 可用前提下，风速问题已经可以优先重写为 `rpm -> wind`；
- 当前 `added` 上最稳的主干并不是深度模型，而是更简单的局部邻域映射。

### 3.2 [2026-04-08] `acc residual` 在 `final LOCO` 上出现了小幅正信号，但必须做保守收缩

`final LOCO` 关键对照：

- `rpm_only__ridge`
  - `case_mae = 0.4053`
- `rpm_ridge__plus__acc_residual_ridge`
  - `case_mae = 0.4034`
- `rpm_ridge__plus__acc_residual_ridge_w0.4`
  - `case_mae = 0.3984`

对比 `rpm_only__ridge`：

- `full residual`
  - `delta_case_mae = -0.0019`
- `residual @ w=0.4`
  - `delta_case_mae = -0.0069`

这说明：

- 加速度修正确实不是完全没用；
- 但最稳的方式不是“让残差全量纠偏”，而是“让残差只做小幅修正”；
- 当前更符合数据规模和任务结构的表达，是：
  - `rpm` 决定主趋势；
  - `acc` 只负责微调。

### 3.3 [2026-04-08] `acc residual` 在 `added external` 上没有带来增益

`added external` 关键对照：

- `rpm_only__knn4`
  - `case_mae = 0.2293`
- `rpm_knn4__plus__acc_residual_ridge_w0.4`
  - `case_mae = 0.3787`
- `rpm_knn4__plus__acc_residual_ridge`
  - `case_mae = 0.6028`

这说明：

- 即使已经把残差权重收缩到 `0.4`，added 上最优结果仍然是“不加修正”；
- 当前加速度修正分支并不能直接升级成 `final + added` 的统一默认附加模块；
- 如果后续继续保留 `acc residual`，更合理的角色应该是：
  - `final` 主域的小修正器；
  - 而不是 added 默认增强器。

## 4. 当前判断

`2026-04-08` 这轮 quickcheck 更支持以下表达：

- 若 `true_rpm` 可用，后续主线应优先切换到 `rpm-first`；
- `TCN` 不再适合做默认主干；
- `acc` 更适合被组织成“受约束的小残差修正分支”，而不是与 `rpm` 平权竞争的主模型；
- 当前已经出现一个值得继续追的正信号：
  - `final LOCO` 上，`rpm_ridge + 0.4 * acc_residual_ridge` 优于纯 `rpm_ridge`；
- 但 added 方向同时给出了一个清楚的反向信号：
  - `added external` 上，当前最佳仍是 `rpm_only__knn4`。

因此，当前更合理的下一步不是继续扩大深度模型，而是优先验证：

- `rpm` 主干是否要分成：
  - `global smooth`（如 `ridge / spline`）
  - `local neighbor`（如 `knn4`）
- `acc residual` 是否应只在 `final-like` 样本上开启；
- 是否需要把这条线进一步重写成：
  - `base = rpm-only`
  - `optional correction = acc residual`
  - `gate = 什么时候允许开修正`

## 5. 一句话版结论

截至 `2026-04-08`，`true_rpm` 一旦可用，当前最值得继续迭代的不是 `TCN` 主干，而是“`rpm-only` 作为主模型，`acc` 只做保守小修正，并且这类修正更像 `final` 主域专用，而不是 added 默认模块”。
