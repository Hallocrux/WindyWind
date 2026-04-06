# 第三阶段 CNN / TCN full 复核（2026-04-05）

- 状态：`current`
- 首次确认：`2026-04-05`
- 最近复核：`2026-04-05`
- 数据范围：manifest 迁移后的 `工况1` 到 `工况20`
- 代码口径：
  - `src/try/013_phase3_cnn_tcn_smoke/`
  - `src/try/012_phase3_end_to_end_shortlist/`
- 证据入口：
  - `outputs/try/013_phase3_cnn_tcn_smoke/cnn_tcn_model_summary.csv`
  - `outputs/try/013_phase3_cnn_tcn_smoke/cnn_tcn_case_level_predictions.csv`

## 1. 目标

在第三阶段用最小可运行配置验证两个轻量端到端模型：

- `Tiny1DCNN`
- `TinyTCN`

并与当前最佳手工特征参考 `TabularReference_G6_Ridge` 做 full 对比。

## 2. 当前结论

### 2.1 [2026-04-05] `TinyTCN` 是当前 full 口径下的最优模型

- 状态：`current`
- 首次确认：`2026-04-05`
- 最近复核：`2026-04-05`

结论：

- `TinyTCN`
  - `case_mae = 0.3030`
  - `case_rmse = 0.4930`
  - `case_mape = 7.9581%`
- `Tiny1DCNN`
  - `case_mae = 0.3340`
  - `case_rmse = 0.5151`
  - `case_mape = 8.0261%`
- `TabularReference_G6_Ridge`
  - `case_mae = 0.4045`
  - `case_rmse = 0.7106`
  - `case_mape = 11.6894%`

因此：

- `TinyTCN` 明显优于当前最佳手工特征参考；
- `Tiny1DCNN` 也优于当前最佳手工特征参考；
- 当前第三阶段的最优候选已经从表格特征模型切换为 `TinyTCN`。

### 2.2 [2026-04-05] `TinyTCN` 相比 `TabularReference_G6_Ridge` 的优势不是单一工况偶然造成

- 状态：`current`
- 首次确认：`2026-04-05`
- 最近复核：`2026-04-05`

结论：

- `TinyTCN` 更好的工况数：`13`
- `TinyTCN` 更差的工况数：`6`
- `TinyTCN` 恶化超过 `0.10 m/s` 的工况数：`2`

改善更明显的工况包括：

- `工况1`
- `工况19`
- `工况16`
- `工况17`
- `工况10`

回退更明显的工况包括：

- `工况18`
- `工况3`

说明：

- `TinyTCN` 不是在所有工况上都更优；
- 但它在更多工况上更优，而且在若干关键工况上的改善幅度足够大，因此整体优势显著。

## 3. 当前判断

`2026-04-05` 的第三阶段 full 复核支持以下判断：

- 当前最值得继续迭代的主线模型是 `TinyTCN`；
- `Tiny1DCNN` 可以保留为轻量深度参考，但不是当前最佳；
- 后续如果继续做第三阶段，应优先围绕 `TinyTCN` 做稳定性、超参数和输入策略复核，而不是回到表格基线做大规模补测。
