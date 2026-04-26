# TCN conservative gate persist（2026-04-08）

- 状态：`current`
- 首次确认：`2026-04-08`
- 最近复核：`2026-04-08`
- 数据范围：
  - `final` 带标签工况 `1, 3-20`
  - `added` 带标签工况 `21-24`
- 代码口径：
  - `src/try/050_tcn_conservative_gate_persist/`
- 证据入口：
  - `outputs/try/050_tcn_conservative_gate_persist/case_level_predictions.csv`
  - `outputs/try/050_tcn_conservative_gate_persist/summary_by_variant.csv`
  - `outputs/try/050_tcn_conservative_gate_persist/summary.md`
  - `outputs/try/050_tcn_conservative_gate_persist/models/checkpoints/`

## 1. 目标

验证把 `TCN gate` 改写成更保守的两阶段结构后，是否能在保持 added 增强收益的同时，更好地压住 final 退化：

1. stage1：
   - 先判断是否允许增强
2. stage2：
   - 若允许，再在 `{0.3, 0.5, 1.0}` 中选择增强权重

## 2. 当前结果

### 2.1 [2026-04-08] `TCN conservative gate` 与 `TCN soft gate` 整体几乎打平

结果：

- `all_labeled | tcn_two_stage`
  - `case_mae = 0.2300`
- `final | tcn_two_stage`
  - `case_mae = 0.2453`
- `added | tcn_two_stage`
  - `case_mae = 0.1571`

对照：

- `049 TCN soft gate`
  - `all_labeled = 0.2301`
  - `final = 0.2567`
  - `added = 0.1036`

这说明：

- 两阶段保守结构在整体 `all_labeled` 上和连续 `TCN soft gate` 基本打平；
- 它在 `final` 上略好于 `049`；
- 但同时失去了 `049` 在 added 上的那部分强恢复。

### 2.2 [2026-04-08] 当前 conservative TCN gate 本质上接近“added 全开增强”

结果：

- `added | tcn_two_stage`
  - `mean_gate = 1.0000`
  - `case_mae = 0.1571`

这说明：

- 当前两阶段 TCN gate 在 added 上几乎退化成“全部走 enhanced expert”；
- 因此它虽然比 `049` 更保守，但并没有学出更细粒度的 added 内部分配。

### 2.3 [2026-04-08] `threshold=0.65` 当前没有带来额外变化

结果：

- `tcn_two_stage`
  - 与
- `tcn_two_stage_t0.65`
  - 指标完全相同

这说明：

- 当前 stage1 的输出已经比较极端；
- 简单再加一个固定阈值并没有改变决策；
- 真正需要的不是更硬的阈值，而是更好的 stage1 监督目标或损失约束。

## 3. 当前判断

截至 `2026-04-08`，`TCN conservative gate` 支持下面的判断：

- 时序 gate 方向是成立的；
- 但“连续 gate”与“两阶段 conservative gate”当前并没有拉开特别大的差距；
- `050` 的主要收益是把 final 稍微拉回一些；
- `049` 的主要收益是把 added 压得更低；
- 当前更像是在 `049` 与 `050` 之间做 trade-off，而不是已经出现绝对优势方案。

## 4. 一句话版结论

截至 `2026-04-08`，`TCN conservative gate` 说明“更保守的两阶段时序 gate”确实能稍微改善 final，但它同时丢掉了一部分 added 收益；当前最真实的结论不是“050 已优于 049”，而是“两个 TCN gate 已经形成了 final/added 的不同偏好，需要下一步继续做显式约束或多目标权衡”。
