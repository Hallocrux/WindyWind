# added 外部验证与可疑标签检查（2026-04-07）

- 状态：`current`
- 首次确认：`2026-04-07`
- 最近复核：`2026-04-07`
- 数据范围：
  - 训练池：`data/final/` 带标签工况，去掉 `工况1 / 3 / 17 / 18`
  - 外部验证池：`data/added/` 的 `工况21-24`
  - 可疑工况：`工况1 / 3 / 17 / 18`
- 代码口径：
  - `src/try/034_added_validation_label_check/`
- 证据入口：
  - `outputs/try/034_added_validation_label_check/added_case_predictions.csv`
  - `outputs/try/034_added_validation_label_check/suspicious_case_predictions.csv`
  - `outputs/try/034_added_validation_label_check/summary.csv`
  - `outputs/try/034_added_validation_label_check/summary.md`

## 1. 目标

使用 `data/added/` 的新补充工况做外部验证，并检查：

- 在去掉当前难工况后，clean training pool 是否还能稳定预测 added 数据；
- 如果可以，再用同一训练池检查可疑工况标签偏离。

## 2. 方法口径

- 训练池：
  - `data/final/` 中带标签工况
  - 显式去掉：
    - `工况1`
    - `工况3`
    - `工况17`
    - `工况18`
- 外部验证：
  - `工况21-24`
- 参考模型：
  - `TinyTCN@2s`
  - `TinyTCN@5s`
  - `TinyTCN@8s`
  - `2s+8s` 工况级晚融合

## 3. 当前结果

### 3.1 [2026-04-07] 当前 clean training pool 不能稳定覆盖 `added 21-24`

- 状态：`current`
- 首次确认：`2026-04-07`
- 最近复核：`2026-04-07`

外部验证结果：

- `工况21`
  - 标签：`4.4`
  - `pred_5s = 6.5895`
  - `pred_fusion = 6.7003`
- `工况22`
  - 标签：`3.4`
  - `pred_5s = 6.5977`
  - `pred_fusion = 6.7035`
- `工况23`
  - 标签：`6.0`
  - `pred_5s = 6.7246`
  - `pred_fusion = 6.8184`
- `工况24`
  - 标签：`5.0`
  - `pred_5s = 6.5793`
  - `pred_fusion = 6.6820`

汇总：

- `added_cases | 5s`
  - `case_mae = 1.9228`
- `added_cases | 2s_8s_fusion`
  - `case_mae = 2.0260`

这说明：

- 当前 clean pool 在 `added` 数据上出现了明显的外部失配；
- 模型几乎把 `21-24` 压到同一个较窄预测带；
- 因此当前不能把 `added 21-24` 当成“已经证明可靠”的外部验证集来审标签。

### 3.2 [2026-04-07] 在当前外部验证失配前提下，对可疑工况的标签检查只能视作弱证据

- 状态：`current`
- 首次确认：`2026-04-07`
- 最近复核：`2026-04-07`

可疑工况结果：

- `工况1`
  - 标签：`2.12`
  - `pred_5s = 4.9003`
  - `pred_fusion = 4.9486`
- `工况3`
  - 标签：`3.11`
  - `pred_5s = 4.3314`
  - `pred_fusion = 4.2427`
- `工况17`
  - 标签：`8.10`
  - `pred_5s = 6.6983`
  - `pred_fusion = 6.7995`
- `工况18`
  - 标签：`5.80`
  - `pred_5s = 4.9981`
  - `pred_fusion = 5.0418`

这些结果仍显示：

- `工况1 / 3` 被明显高估；
- `工况17 / 18` 被低估；

但由于 added 外部验证本身已经不稳，当前不能直接把这些偏差解释成“标签错误”的强证据。

## 4. 当前判断

`2026-04-07` 的这轮检查支持以下判断：

- `data/added/` 当前更像是一个额外分布，而不是可以直接拿来做标签审计的稳定外部验证集；
- 在没有先解释 `added 21-24` 的分布偏移前，不应直接用它来判定 `工况1 / 18` 标签是否错误；
- 当前更合理的下一步是：
  - 先把 `added 21-24` 放进机制特征空间里看它们与主数据的距离；
  - 再决定它们是否适合作为标签审计验证集。
