# added 外部验证（包含难工况训练池，2026-04-07）

- 状态：`current`
- 首次确认：`2026-04-07`
- 最近复核：`2026-04-07`
- 数据范围：
  - 训练池：`data/final/` 全部带标签工况
  - 外部验证池：`data/added/` 的 `工况21-24`
- 代码口径：
  - `src/try/035_added_validation_with_full_final_pool/`
- 证据入口：
  - `outputs/try/035_added_validation_with_full_final_pool/added_case_predictions.csv`
  - `outputs/try/035_added_validation_with_full_final_pool/summary.csv`
  - `outputs/try/035_added_validation_with_full_final_pool/summary.md`

## 1. 目标

在 `034` 已证明 `added` 外部验证不稳定的前提下，做一个最小对照：

- 把难工况重新加入训练池；
- 再看 `added 21-24` 的外部预测是否改善。

## 2. 方法口径

- 训练池：
  - `data/final/` 中全部带标签工况
  - 不再去掉 `工况1 / 3 / 17 / 18`
- 外部验证：
  - `工况21-24`
- 模型：
  - `TinyTCN@2s`
  - `TinyTCN@5s`
  - `TinyTCN@8s`
  - `2s+8s` 工况级晚融合

## 3. 当前结果

### 3.1 [2026-04-07] 把难工况加回训练池后，`added` 外部验证进一步恶化

- 状态：`current`
- 首次确认：`2026-04-07`
- 最近复核：`2026-04-07`

`added` 预测结果：

- `工况21`
  - 标签：`4.4`
  - `pred_5s = 7.8197`
  - `pred_fusion = 7.4864`
- `工况22`
  - 标签：`3.4`
  - `pred_5s = 8.2063`
  - `pred_fusion = 7.6412`
- `工况23`
  - 标签：`6.0`
  - `pred_5s = 7.4192`
  - `pred_fusion = 7.3599`
- `工况24`
  - 标签：`5.0`
  - `pred_5s = 7.9458`
  - `pred_fusion = 7.5314`

汇总：

- `5s`
  - `case_mae = 3.1478`
- `2s+8s fusion`
  - `case_mae = 2.8047`

对比 `034`：

- `5s`
  - 从 `1.9228` 恶化到 `3.1478`
- `2s+8s fusion`
  - 从 `2.0260` 恶化到 `2.8047`

## 4. 当前判断

`2026-04-07` 的这轮对照支持以下判断：

- `added 21-24` 与当前主训练池之间确实存在明显分布距离；
- 把难工况重新加入训练池并不能修复这件事，反而会进一步拉高对 `added` 的系统性高估；
- 因此当前不应把 “难工况污染训练池” 视为 `added` 预测失配的主解释。
