# added+added2 训练池的 TinyTCN `2s+8s` 晚融合少量 LOCO quickcheck（2026-04-09）

- 状态：`current`
- 首次确认：`2026-04-09`
- 最近复核：`2026-04-09`
- 数据范围：
  - `data/added/` 的带标签工况 `21-24`
  - `data/added2/` 的带标签工况 `25-30`
- 代码口径：
  - `src/try/085_added_added2_late_fusion_loco_quickcheck/`
- 证据入口：
  - `outputs/try/085_added_added2_late_fusion_loco_quickcheck/case_level_predictions.csv`
  - `outputs/try/085_added_added2_late_fusion_loco_quickcheck/summary_by_domain.csv`
  - `outputs/try/085_added_added2_late_fusion_loco_quickcheck/summary.md`
  - `outputs/try/085_added_added2_late_fusion_loco_quickcheck/run_config.json`

## 1. 目标

做一个尽快出结果的小规模验证：

- 不再用 `final`；
- 只把 `added + added2` 的带标签工况作为训练池；
- 直接复用现有 `TinyTCN@2s`、`TinyTCN@8s` 与工况级 `2s+8s` 晚融合；
- 只挑少量代表性工况做 case-level `LOCO`。

这轮想回答的问题是：

- 当训练池切换成 `added + added2` 之后，direct learned `TinyTCN 2s+8s` 是否会比单独 `2s` 更稳；
- 这个小外部池内部，误差模式更像“短窗优先”还是“多尺度互补”。

## 2. 方法口径

- 统一训练池：
  - `added 21-24`
  - `added2 25-30`
  - 总计 `10` 个带标签工况
- 输入列：
  - 取 `added + added2` 的共同有效信号列，共 `20` 列
- 清洗逻辑：
  - 复用 `src/current.data_loading`
- 窗口：
  - `2s`
  - `8s`
- 融合：
  - `pred_fusion = 0.5 * pred_2s + 0.5 * pred_8s`
- 训练参数：
  - `max_epochs = 20`
  - `patience = 4`
  - 单 seed quickcheck
- 本轮 holdout：
  - `工况22`：`added` 低风速 hard case
  - `工况23`：`added` 高风速代表
  - `工况25`：`added2` 高风速代表
  - `工况29`：`added2` 低风速代表

## 3. 当前结果

### 3.1 [2026-04-09] 在 `added + added2` 的少量 LOCO quickcheck 上，`2s` 仍然优于 `2s+8s` 与 `8s`

- 状态：`current`
- 首次确认：`2026-04-09`
- 最近复核：`2026-04-09`

汇总如下：

- `added`
  - `2s`
    - `case_mae = 1.5115`
  - `2s+8s fusion`
    - `case_mae = 1.5625`
  - `8s`
    - `case_mae = 1.6135`
- `added2`
  - `2s`
    - `case_mae = 1.1139`
  - `2s+8s fusion`
    - `case_mae = 1.3008`
  - `8s`
    - `case_mae = 1.4878`
- `all_holdouts`
  - `2s`
    - `case_mae = 1.3127`
  - `2s+8s fusion`
    - `case_mae = 1.4317`
  - `8s`
    - `case_mae = 1.5506`

这说明：

- 至少在这轮少量 holdout quickcheck 上，训练池改成 `added + added2` 后，`2s+8s` 仍没有成为最优；
- `2s` 在 `added`、`added2` 和总 holdout 三桶里都排第一；
- `8s` 在这轮里 consistently 最弱。

### 3.2 [2026-04-09] 当前误差模式表现为“added 高估、added2 一高一低混合”，但晚融合没有形成稳定互补

- 状态：`current`
- 首次确认：`2026-04-09`
- 最近复核：`2026-04-09`

每工况预测如下：

- `added | 工况22`
  - `true = 3.4000`
  - `pred_2s = 4.2143`
  - `pred_8s = 4.4020`
  - `pred_fusion = 4.3081`
  - `fusion abs_error = 0.9081`
- `added | 工况23`
  - `true = 6.0000`
  - `pred_2s = 8.2087`
  - `pred_8s = 8.2250`
  - `pred_fusion = 8.2168`
  - `fusion abs_error = 2.2168`
- `added2 | 工况25`
  - `true = 8.5000`
  - `pred_2s = 6.8333`
  - `pred_8s = 5.9777`
  - `pred_fusion = 6.4055`
  - `fusion abs_error = 2.0945`
- `added2 | 工况29`
  - `true = 3.6000`
  - `pred_2s = 4.1611`
  - `pred_8s = 4.0533`
  - `pred_fusion = 4.1072`
  - `fusion abs_error = 0.5072`

这说明：

- `added` 两个 holdout 在这轮里都表现为高估；
- `added2` 则不是单一方向崩坏，而是高风速 `工况25` 明显低估、低风速 `工况29` 轻度高估；
- `2s` 与 `8s` 的偏差方向并没有形成足够强的互补，因此简单 `0.5/0.5` 晚融合没有带来增益。

## 4. 当前判断

截至 `2026-04-09`，这轮 quickcheck 更支持下面的临时判断：

- 把训练池切成 `added + added2` 之后，direct learned `TinyTCN` 并没有自动回到“多尺度更优”的状态；
- 当前更像是：
  - 小外部池内部仍然更依赖 `2s` 短窗；
  - `8s` 提供的补充信号还不足以抵消它带来的偏差；
- 因此如果后续还要继续追 `added + added2` 内部 direct learned 主干，更值得优先做：
  - `2s` 单窗复核
  - 保守加权而不是固定 `0.5/0.5`
  - 或只在特定 case bucket 上打开 `8s`

而不是直接把当前 `2s+8s` 晚融合升级成默认候选。
