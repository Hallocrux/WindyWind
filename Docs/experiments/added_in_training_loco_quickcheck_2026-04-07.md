# added 并入训练池后的 2s/8s 统一 LOCO quickcheck（2026-04-07）

- 状态：`current`
- 首次确认：`2026-04-07`
- 最近复核：`2026-04-07`
- 数据范围：
  - `data/final/` 的带标签工况 `1, 3-20`
  - `data/added/` 的带标签工况 `21-24`
- 代码口径：
  - `src/try/045_added_in_training_loco_quickcheck/`
- 证据入口：
  - `outputs/try/045_added_in_training_loco_quickcheck/case_level_predictions.csv`
  - `outputs/try/045_added_in_training_loco_quickcheck/summary_by_domain.csv`
  - `outputs/try/045_added_in_training_loco_quickcheck/summary.md`

## 1. 目标

做一个最小成本的信号验证：

- 不做 FFT、midband、route/gate；
- 只复用现有 `TinyTCN@2s`、`TinyTCN@8s` 与工况级 `2s+8s` 晚融合；
- 把 `final` 与 `added` 合成统一带标签池；
- 在统一池上做 case-level `LOCO`；
- 分别汇报：
  - `final` 子集
  - `added` 子集
  - `all_labeled` 整体

本 try 的问题是：

- 如果不把 `added` 当外部域，而是直接并入训练池，当前 `2s/8s` 主干会不会出现值得继续追的正信号。

## 2. 方法口径

- 训练池：
  - `final` 带标签工况 `19` 个
  - `added` 带标签工况 `4` 个
  - 总计 `23` 个带标签工况
- 清洗逻辑：
  - 复用 `src/current.data_loading`
- 输入列口径：
  - 取 `final + added` 共有有效传感器列
- 评估方式：
  - 对 `23` 个带标签工况做统一 case-level `LOCO`
- 模型：
  - `TinyTCN@2s`
  - `TinyTCN@8s`
  - `2s+8s` 工况级晚融合
- 融合定义：
  - `pred_fusion = 0.5 * pred_2s + 0.5 * pred_8s`
- 随机性：
  - 单 seed quickcheck

## 3. 当前结果

### 3.1 [2026-04-07] 在统一 `final + added` LOCO 下，当前单 seed 最优变成了 `2s`，`2s+8s` 没有保住原先优势

- 状态：`current`
- 首次确认：`2026-04-07`
- 最近复核：`2026-04-07`

三桶汇总如下：

- `final`
  - `2s`
    - `case_mae = 0.5034`
  - `2s+8s fusion`
    - `case_mae = 0.5143`
  - `8s`
    - `case_mae = 0.5429`
- `added`
  - `2s`
    - `case_mae = 0.6604`
  - `2s+8s fusion`
    - `case_mae = 0.7273`
  - `8s`
    - `case_mae = 0.7942`
- `all_labeled`
  - `2s`
    - `case_mae = 0.5307`
  - `2s+8s fusion`
    - `case_mae = 0.5513`
  - `8s`
    - `case_mae = 0.5866`

这说明至少在当前单 seed quickcheck 里：

- 把 `added` 并入训练池后，`2s+8s` 并没有成为统一池最优；
- `2s` 短窗反而在三桶里都排第一；
- 原来在 `final` 主域中成立的 `2s+8s` 强互补信号，没有自然迁移成“并入 added 后仍稳定更优”的统一主线证据。

### 3.2 [2026-04-07] added 并入训练后，`added` 子集确实比此前“只用 final 训练再外推 added”明显改善

- 状态：`current`
- 首次确认：`2026-04-07`
- 最近复核：`2026-04-07`

与 `035` 的外部验证相比：

- `035 | added external | 2s+8s fusion`
  - `case_mae = 2.8047`
- `045 | unified LOCO | added subset | 2s+8s fusion`
  - `case_mae = 0.7273`
- `045 | unified LOCO | added subset | 2s`
  - `case_mae = 0.6604`

这说明：

- `added` 一旦进入训练池并按统一 `LOCO` 评估，确实能从原先严重外推失配中明显恢复；
- 所以“added 完全学不会”并不是当前更准确的结论；
- 更准确的问题变成了：
  - `added` 加入训练后，能否在提升 `added` 的同时保住原 `final` 主干的结构性优势。

### 3.3 [2026-04-07] 当前 unified quickcheck 仍暴露出若干 final hardest case 未被修平

- 状态：`current`
- 首次确认：`2026-04-07`
- 最近复核：`2026-04-07`

单工况结果里仍有明显长尾：

- `工况1`
  - `2s abs_error = 2.9851`
  - `2s+8s fusion abs_error = 2.8689`
- `工况17`
  - `2s abs_error = 1.5650`
  - `2s+8s fusion abs_error = 1.6528`
- `工况18`
  - `2s abs_error = 1.0423`
  - `2s+8s fusion abs_error = 1.0359`
- `added` 中最难点仍是 `工况22`
  - `2s abs_error = 1.1191`
  - `2s+8s fusion abs_error = 1.2189`

这说明：

- 当前 unified quickcheck 并没有把 hardest cases 自动一起修平；
- 它更像是在告诉我们：
  - `added` 加入训练池有帮助；
  - 但“直接把 added 拼进来 + 原样复用 2s/8s 融合”还不足以形成新的统一默认主线。

## 4. 当前判断

截至 `2026-04-07`，这轮 quickcheck 更支持下面的临时判断：

- `added` 并入训练池后，`added` 子集会明显受益；
- 但当前单 seed 下，统一池最优不是 `2s+8s`，而是更简单的 `2s`；
- 因此这个方向有“可以继续追”的信号，但信号不是“`2s+8s` 统一主线已经成立”，而是：
  - `added` 进入训练确实改变了可学性；
  - 但统一主线的最佳结构可能已经偏离原先 `final` 域最优配方。

## 5. 下一步建议

如果继续保持“小实验先拿信号”的节奏，更推荐按下面顺序推进：

1. 先对这轮 unified `LOCO` 做 `2s / 5s / 8s / 2s+8s` 的同口径小矩阵补齐，确认 `2s` 领先是不是单次偶然。
2. 再补一个少量 seed 复核，先看：
   - `2s`
   - `2s+8s fusion`
3. 若 `2s` 持续领先，再决定：
   - 是重做 unified 多尺度权重
   - 还是承认 `added` 并入后最优窗口结构已经变化。
