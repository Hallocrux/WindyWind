# TinyTCN 第一优先级快速验证（2026-04-06）

- 状态：`current`
- 首次确认：`2026-04-06`
- 最近复核：`2026-04-06`
- 数据范围：
  - 目标难工况：`工况1 / 3 / 17 / 18`
  - 多尺度补充复核：带标签工况 `19` 个
- 代码口径：
  - `src/try/026_tinytcn_priority1_quickcheck/`
- 证据入口：
  - `outputs/try/026_tinytcn_priority1_quickcheck/variant_case_level_comparison.csv`
  - `outputs/try/026_tinytcn_priority1_quickcheck/variant_summary.csv`
  - `outputs/try/026_tinytcn_priority1_quickcheck/full19_multiscale_late_fusion_2s_8s_case_level.csv`
  - `outputs/try/026_tinytcn_priority1_quickcheck/full19_multiscale_late_fusion_2s_8s_summary.csv`

## 1. 目标

快速验证第一优先级中的两个方向是否值得继续投入：

- 方向 `1`：多尺度 `TinyTCN`
- 方向 `3`：按工况均衡加权训练

为了控制成本，这次不扩成大矩阵：

- 对方向 `1`，先用“多窗长晚融合”代理多尺度思路；
- 对方向 `3`，只验证 `5s TinyTCN + inverse case-window-count weighting`；
- 难工况只看 `工况1 / 3 / 17 / 18`；
- 多尺度方向另外补一份不需要训练的 `19` 工况全量复核。

## 2. 方法口径

### 2.1 多尺度方向

- 直接复用 `014` 已有结果：
  - `TinyTCN@2s`
  - `TinyTCN@5s`
  - `TinyTCN@8s`
- 构造多尺度代理：
  - `TinyTCN_multiscale_late_fusion_2s_8s`
  - 工况级预测值定义为：`0.5 * pred_2s + 0.5 * pred_8s`

### 2.2 工况均衡加权方向

- 模型：`TinyTCN@5s`
- 只改训练损失：
  - 训练窗口权重按所属工况窗口数倒数设置；
  - 使每个训练工况在总 loss 中贡献更接近相同。
- 快速训练配置：
  - `max_epochs = 24`
  - `patience = 5`

## 3. 当前结果

### 3.1 [2026-04-06] 多尺度晚融合在目标难工况上给出强正信号

- 状态：`current`
- 首次确认：`2026-04-06`
- 最近复核：`2026-04-06`

目标难工况 `工况1 / 3 / 17 / 18` 的结果：

- `TinyTCN@5s`
  - `case_mae = 0.9955`
- `TinyTCN_multiscale_late_fusion_2s_8s`
  - `case_mae = 0.2985`

逐工况变化：

- `工况1`
  - 从 `1.1989` 降到 `0.4387`
- `工况3`
  - 从 `1.0515` 降到 `0.0425`
- `工况17`
  - 从 `1.0883` 降到 `0.0862`
- `工况18`
  - 从 `0.6433` 降到 `0.6267`

这说明：

- 至少在当前最难的几类工况上，短窗与长窗的信息确实具有互补性；
- 多尺度方向值得继续做成真正的联合模型，而不只是单窗长扫描。

### 3.2 [2026-04-06] 多尺度晚融合在 `19` 工况全量复核上也保持明显优势

- 状态：`current`
- 首次确认：`2026-04-06`
- 最近复核：`2026-04-06`

全量 `19` 工况结果：

- `TinyTCN@5s`
  - `case_mae = 0.3030`
  - `case_rmse = 0.4930`
  - `case_max_error = 1.1989`
- `TinyTCN_multiscale_late_fusion_2s_8s`
  - `case_mae = 0.1685`
  - `case_rmse = 0.2522`
  - `case_max_error = 0.6267`

工况级统计：

- 改善工况数：`10 / 19`
- 恶化工况数：`9 / 19`

虽然不是所有工况都更好，但在少数难工况上的改善幅度很大，整体指标明显下移。

### 3.3 [2026-04-06] 按工况均衡加权训练没有表现出稳定正信号

- 状态：`current`
- 首次确认：`2026-04-06`
- 最近复核：`2026-04-06`

难工况 `工况1 / 3 / 17 / 18` 的结果：

- `TinyTCN@5s`
  - `case_mae = 0.9955`
- `TinyTCN_5s_case_balanced`
  - `case_mae = 0.8290`

这条线并非完全无效，因为：

- `工况3`：`1.0515 -> 0.3105`
- `工况17`：`1.0883 -> 1.0216`

但它也带来回退：

- `工况1`：`1.1989 -> 1.2125`
- `工况18`：`0.6433 -> 0.7713`

补充对照工况 `工况6 / 8 / 10 / 13` 上，`case-balanced` 相比 `5s baseline` 还会整体变差：

- `TinyTCN@5s`
  - `case_mae = 0.0066`
- `TinyTCN_5s_case_balanced`
  - `case_mae = 0.0730`

这说明当前这版简单加权还不够稳，不适合直接作为主线升级。

## 4. 当前判断

`2026-04-06` 的这轮快速验证支持以下判断：

- 第一优先级里的方向 `1` 有强可行性信号，应优先继续推进；
- 下一步更合理的是实现真正的多尺度联合模型，而不是只保留后处理晚融合；
- 第一优先级里的方向 `3` 暂时不应单独上主线；
- 如果后续还想保留方向 `3`，更适合把它作为多尺度模型里的一个可选训练策略，而不是单独先做默认升级。
