# 线性表格基线 added-first 补测（2026-04-09）

- 状态：`current`
- 首次确认：`2026-04-09`
- 最近复核：`2026-04-09`
- 数据范围：
  - `data/added/` 的带标签工况 `21-24`
  - `data/added2/` 的带标签工况 `25-30`
- 代码口径：
  - `src/try/080_tabular_linear_baselines_added_first_eval/`
- 证据入口：
  - `outputs/try/080_tabular_linear_baselines_added_first_eval/all_case_predictions.csv`
  - `outputs/try/080_tabular_linear_baselines_added_first_eval/summary_by_protocol.csv`
  - `outputs/try/080_tabular_linear_baselines_added_first_eval/summary_by_protocol_and_domain.csv`
  - `outputs/try/080_tabular_linear_baselines_added_first_eval/summary.md`

## 1. 目标

补回答一个此前没有在 added-first 协议下单独验证的问题：

- 早期线性最强参考 `G6`；
- 以及正式表格主线里的 `Ridge + VIB_FT_RPM`；

放到当前同一 external-first 协议下时，是否会对 `071` 的最终结论构成挑战。

本轮统一对照：

- `TabularReference_G6_Ridge`
- `Ridge + VIB_FT_RPM`
- `rpm_knn4`
- `071 | rpm_knn4 + embedding residual ridge`

## 2. 方法口径

- `TabularReference_G6_Ridge`
  - 复用 `009/012` 的 `G6_TIME_FREQ_CROSS`
  - 模型固定为 `StandardScaler + Ridge(alpha=1.0)`
  - 不使用 `rpm`
- `Ridge + VIB_FT_RPM`
  - 复用正式 `src/current/features.py` 的 `VIB_FT` 特征
  - 在其上追加 `rpm`
  - 模型固定为 `StandardScaler + Ridge(alpha=1.0)`
- 评估口径：
  - 主口径：`added_to_added2`
  - 次口径：`external_loocv`
- 预测方式：
  - 先做窗口级预测；
  - 再按工况平均成 case-level 结果。

## 3. 当前结果

### 3.1 [2026-04-09] 主口径下，`071` 仍然显著领先两条线性表格基线

`added_to_added2` 结果：

- `071 | rpm_knn4 + embedding residual ridge`
  - `case_mae = 0.6161`
- `rpm_knn4`
  - `case_mae = 1.2903`
- `Ridge + VIB_FT_RPM`
  - `case_mae = 1.4522`
- `TabularReference_G6_Ridge`
  - `case_mae = 1.9618`

这说明：

- `G6` 补测后并没有接近 `071`；
- `Ridge + VIB_FT_RPM` 也没有形成 added-first 主口径上的竞争力；
- `071` 的最终结论不受这次补测影响。

### 3.2 [2026-04-09] 次口径下，`Ridge + VIB_FT_RPM` 只显示出弱参考价值

`external_loocv` 结果：

- `rpm_knn4`
  - `case_mae = 0.7772`
- `Ridge + VIB_FT_RPM`
  - `case_mae = 0.8273`
- `071 | rpm_knn4 + embedding residual ridge`
  - `case_mae = 0.8451`
- `TabularReference_G6_Ridge`
  - `case_mae = 1.2185`

这说明：

- 在 external family 内部整体 `LOOCV` 上，`Ridge + VIB_FT_RPM` 不是完全无效；
- 但它的优势只停留在次口径，而且没有迁移成 `added_to_added2` 主口径优势；
- 因此它不能替代 `071` 成为 added-first 的默认最终模型。

### 3.3 [2026-04-09] `G6` 的问题主要出在 `added_to_added2` 的低风速 added2 端明显高估

`TabularReference_G6_Ridge` 在 `added_to_added2` 的 case 级结果：

- `工况28`
  - `pred = 7.4517`
  - `abs_error = 3.7517`
- `工况29`
  - `pred = 7.1420`
  - `abs_error = 3.5420`
- `工况30`
  - `pred = 5.2255`
  - `abs_error = 1.9255`

这说明：

- `G6` 在当前 external-first 场景下，明显会把低风速 added2 case 往高风速侧拉；
- 它更像保留了旧表格域里的统计结构；
- 但没有形成 added 系列外部域上的稳定迁移能力。

### 3.4 [2026-04-09] `Ridge + VIB_FT_RPM` 比 `G6` 强，但仍弱于 `rpm_knn4`

`added_to_added2` 对照：

- `rpm_knn4`
  - `case_mae = 1.2903`
- `Ridge + VIB_FT_RPM`
  - `case_mae = 1.4522`
- `TabularReference_G6_Ridge`
  - `case_mae = 1.9618`

这说明：

- 如果只在两条线性表格基线里比较，正式 `VIB_FT_RPM` 明显优于早期 `G6`；
- 但即使是 `VIB_FT_RPM`，在 added-first 主口径下也仍不如简单的 `rpm_knn4`；
- 因此 added-first 方向当前并不存在一个被漏掉的强线性表格挑战者。

## 4. 当前判断

截至 `2026-04-09`，本轮补测支持下面的收束表达：

- `G6` 此前确实没有在 added-first 协议下单独补测；
- 现在补测后，可以明确排除它作为 added-first 默认主候选的可能性；
- `Ridge + VIB_FT_RPM` 也没有在主口径上超过 `rpm_knn4`；
- 因而 added-first 子项目“最终结论就是 `071`”这件事，现在比补测前更完整。

## 5. 一句话版结论

截至 `2026-04-09`，added-first 子项目里此前漏掉的线性表格基线已经补测完：`G6` 明显不行，`Ridge + VIB_FT_RPM` 也没有超过 `rpm_knn4`，因此 `071 | rpm_knn4 + embedding residual ridge` 仍然是当前无争议的默认最终模型。
