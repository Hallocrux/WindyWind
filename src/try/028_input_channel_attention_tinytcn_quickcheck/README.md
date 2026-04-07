# 028 输入通道注意力 TinyTCN 快速验证

## 目标

- 快速验证“单流 TinyTCN + 轻量输入通道注意力”是否值得继续投入；
- 保持其余条件尽量不变，只比较：
  - `SingleStreamTinyTCN`
  - `InputChannelAttentionTinyTCN`
- 先看目标难工况：
  - `工况1`
  - `工况3`
  - `工况17`
  - `工况18`
- 如果目标难工况上出现稳定正信号，再自动补一组容易工况对照：
  - `工况6`
  - `工况8`
  - `工况10`
  - `工况13`

## 输入与口径

- 数据入口：
  - `data/final/dataset_manifest.csv`
  - `data/final/datasets/工况*.csv`
- 清洗逻辑：复用 `src/current.data_loading`
- 原始窗口构造：复用 `src/try/012_phase3_end_to_end_shortlist/`
- 模型与训练：
  - 保持 `50Hz / 5s / 2.5s`
  - 按工况留一 `LOCO`
  - 只做目标工况折，不重跑全量 `19` 折

## 注意力设计

- 在原始 `20` 通道输入上做一次轻量 `Squeeze-and-Excitation`：
  - 先沿时间维做全局平均
  - 再用两层小 MLP 产出每个通道的权重
  - 最后把权重乘回原始输入
- 后续仍然接当前单流 `TinyTCN` 主体

## 运行方式

```powershell
.venv\Scripts\python.exe src/try/028_input_channel_attention_tinytcn_quickcheck/run_input_channel_attention_tinytcn_quickcheck.py
```

## 输出

- 输出目录：`outputs/try/028_input_channel_attention_tinytcn_quickcheck/`
- 固定产物：
  - `target_case_variant_comparison.csv`
  - `target_case_variant_summary.csv`
  - `control_case_variant_comparison.csv`
  - `control_case_variant_summary.csv`
  - `summary.md`
