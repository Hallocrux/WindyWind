# 双流 TinyTCN 快速验证（2026-04-07）

- 状态：`current`
- 首次确认：`2026-04-07`
- 最近复核：`2026-04-07`
- 数据范围：
  - 目标难工况：`工况1 / 3 / 17 / 18`
- 代码口径：
  - `src/try/027_dualstream_tinytcn_quickcheck/`
- 证据入口：
  - `outputs/try/027_dualstream_tinytcn_quickcheck/target_case_variant_comparison.csv`
  - `outputs/try/027_dualstream_tinytcn_quickcheck/target_case_variant_summary.csv`
  - `outputs/try/027_dualstream_tinytcn_quickcheck/summary.md`

## 1. 目标

快速验证“把输入改成双流 TinyTCN”是否值得继续投入：

- `SingleStreamTinyTCN`
- `DualStreamTinyTCN`

这里的“双流”定义为：

- `strain` 通道单独编码
- `acc` 通道单独编码
- 两路表示在 head 前拼接

## 2. 方法口径

- 数据入口：`data/final/dataset_manifest.csv` 与 `data/final/datasets/工况*.csv`
- 清洗逻辑：复用 `src/current.data_loading`
- 切窗口径：
  - `50Hz`
  - `5s`
  - `2.5s`
- 评估方式：
  - `Leave-One-Condition-Out`
  - 只重跑目标 `4` 个 eval 工况折
- 通道分组：
  - `strain = 5`
  - `acc = 15`
- 快速训练配置：
  - `max_epochs = 24`
  - `patience = 5`

## 3. 当前结果

### 3.1 [2026-04-07] 当前这版双流 TinyTCN 没有给出正信号

- 状态：`current`
- 首次确认：`2026-04-07`
- 最近复核：`2026-04-07`

目标难工况 `工况1 / 3 / 17 / 18` 的结果：

- `SingleStreamTinyTCN`
  - `case_mae = 0.6566`
  - `case_max_error = 0.8964`
- `DualStreamTinyTCN`
  - `case_mae = 1.1937`
  - `case_max_error = 1.8811`

逐工况变化：

- `工况1`
  - `0.4205 -> 1.8811`
- `工况3`
  - `0.8964 -> 0.9666`
- `工况17`
  - `0.4467 -> 1.3011`
- `工况18`
  - `0.8629 -> 0.6260`

双流只在 `工况18` 上更好，其余 `3` 个目标工况都更差。

### 3.2 [2026-04-07] 这轮快验不支持继续扩到控制工况

- 状态：`current`
- 首次确认：`2026-04-07`
- 最近复核：`2026-04-07`

这轮验证预设的继续条件是：

- 双流在目标难工况上的 `case_mae` 低于单流；
- 且至少改善 `3 / 4` 个目标工况。

当前结果不满足该条件，因此：

- 没有继续扩到 `工况6 / 8 / 10 / 13` 控制工况；
- 本轮结论应直接停在“当前这版双流结构无正信号”。

## 4. 当前判断

`2026-04-07` 的这轮快速验证支持以下判断：

- “按传感器类型硬拆成双流、在 head 前再拼接”这版实现，不适合作为当前主线升级；
- 当前仓库里更值得继续投入的仍然是“多尺度”方向，而不是“双流”方向；
- 如果未来还想再看双流，应该把它放到更后面的候选里，并与多尺度联合设计，而不是单独先推进。
