# 后卷积通道注意力 TinyTCN 快速验证（2026-04-07）

- 状态：`current`
- 首次确认：`2026-04-07`
- 最近复核：`2026-04-07`
- 数据范围：
  - 目标难工况：`工况1 / 3 / 17 / 18`
- 代码口径：
  - `src/try/029_postconv_channel_attention_tinytcn_quickcheck/`
- 证据入口：
  - `outputs/try/029_postconv_channel_attention_tinytcn_quickcheck/target_case_variant_comparison.csv`
  - `outputs/try/029_postconv_channel_attention_tinytcn_quickcheck/target_case_variant_summary.csv`
  - `outputs/try/029_postconv_channel_attention_tinytcn_quickcheck/summary.md`

## 1. 目标

快速验证“先卷积提取时序特征，再对 hidden channels 做轻量 attention”是否值得继续投入：

- `SingleStreamTinyTCN`
- `PostConvChannelAttentionTinyTCN`

这里的 attention 定义为：

- 先经过当前 `TinyTCN` 编码块得到 hidden feature maps；
- 再沿时间维做全局平均池化；
- 用两层小 MLP 产出 hidden channels 的权重；
- 把权重乘回 hidden feature maps 后，再做全局池化与回归。

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
- 快速训练配置：
  - `max_epochs = 20`
  - `patience = 4`

## 3. 当前结果

### 3.1 [2026-04-07] 当前这版后卷积通道注意力没有给出正信号

- 状态：`current`
- 首次确认：`2026-04-07`
- 最近复核：`2026-04-07`

目标难工况 `工况1 / 3 / 17 / 18` 的结果：

- `SingleStreamTinyTCN`
  - `case_mae = 0.5387`
  - `case_max_error = 0.8629`
- `PostConvChannelAttentionTinyTCN`
  - `case_mae = 0.7325`
  - `case_max_error = 1.1538`

逐工况变化：

- `工况1`
  - `0.4205 -> 0.0273`
- `工况3`
  - `0.4247 -> 1.0843`
- `工况17`
  - `0.4467 -> 1.1538`
- `工况18`
  - `0.8629 -> 0.6647`

后卷积注意力改善了 `工况1` 与 `工况18`，但在 `工况3` 与 `工况17` 上明显变差。

### 3.2 [2026-04-07] 本轮结果不支持继续扩到控制工况

- 状态：`current`
- 首次确认：`2026-04-07`
- 最近复核：`2026-04-07`

这轮验证预设的继续条件是：

- attention 版本在目标难工况上的 `case_mae` 低于单流；
- 且至少改善 `3 / 4` 个目标工况。

当前结果不满足该条件，因此：

- 没有继续扩到 `工况6 / 8 / 10 / 13` 控制工况；
- 本轮结论应直接停在“当前这版后卷积 attention 无正信号”。

## 4. 当前判断

`2026-04-07` 的这轮快速验证支持以下判断：

- 相比输入端 gate，这版“先卷积后 attention”更贴近标准 SE 思路；
- 但在当前目标难工况上，仍然没有给出稳定正信号；
- 当前主线里最值得继续投入的仍然是多尺度方向，而不是单独的通道 attention 方向。
