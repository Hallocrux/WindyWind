# TCN soft gate persist（2026-04-08）

- 状态：`current`
- 首次确认：`2026-04-08`
- 最近复核：`2026-04-08`
- 数据范围：
  - `final` 带标签工况 `1, 3-20`
  - `added` 带标签工况 `21-24`
- 代码口径：
  - `src/try/049_tcn_soft_gate_persist/`
- 证据入口：
  - `outputs/try/049_tcn_soft_gate_persist/case_level_predictions.csv`
  - `outputs/try/049_tcn_soft_gate_persist/summary_by_variant.csv`
  - `outputs/try/049_tcn_soft_gate_persist/summary.md`
  - `outputs/try/049_tcn_soft_gate_persist/models/checkpoints/`

## 1. 目标

验证如果把 gate 从表格小模型升级到 `TinyTCN`，并直接输入原始多通道时序窗口，是否能更好地学习连续融合比例：

- `pred = (1 - g) * pred_base + g * pred_enhanced`

同时把每个 fold 与全量 deploy 模型都持久化保存，便于后续复用。

## 2. 当前结果

### 2.1 [2026-04-08] `TCN soft gate` 明显优于此前表格 gate

结果：

- `all_labeled`
  - `case_mae = 0.2301`
- `final`
  - `case_mae = 0.2567`
- `added`
  - `case_mae = 0.1036`

对照：

- `047 hgb_gate`
  - `all_labeled = 0.3219`
  - `final = 0.2366`
  - `added = 0.7273`
- `048 two_stage_hgb`
  - `all_labeled = 0.2931`
  - `final = 0.2552`
  - `added = 0.4733`

这说明：

- 把 gate 升级成时序模型后，added 方向收益恢复得非常明显；
- `TCN soft gate` 是当前 gate 家族里第一个把 `added` 拉回到很低误差区间的候选；
- 它在整体 `all_labeled` 上也已经优于此前所有表格 gate。

### 2.2 [2026-04-08] `TCN soft gate` 仍未压回 `base_only`

对照：

- `base_only | final`
  - `case_mae = 0.1685`
- `TCN soft gate | final`
  - `case_mae = 0.2567`

这说明：

- 当前最大的未解决问题仍然是 `final` 保护不够；
- `TCN soft gate` 已经说明“时序 gate 比表格 gate 更能抓住 added 增强信号”；
- 但它还没有解决“哪些 final 工况绝对不该开增强”的问题。

## 3. 当前判断

截至 `2026-04-08`，`TCN soft gate` 支持下面的判断：

- 如果 gate 确实要看时序，`TinyTCN` 比当前这版表格 gate 更有希望；
- 当前 gate 方向的主要收益来自 added 侧恢复，而不是 final 侧保护；
- 后续更值得继续验证的是：
  - 给 `TCN gate` 加更强的 final 保护约束
  - 或把它与更保守的两阶段逻辑结合，而不是单独自由回归 `g`

## 4. 一句话版结论

截至 `2026-04-08`，`TCN soft gate` 已经显著优于此前的表格 gate，并把 added 误差压到了很低水平，但 final 退化仍未解决，因此它更像“强增强候选”，还不是可直接替代默认主线的最终 gate。
