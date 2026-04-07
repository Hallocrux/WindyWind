# 027 双流 TinyTCN 快速验证

## 目标

- 快速验证把输入改成“双流 TinyTCN”是否值得继续投入；
- 保持其余条件尽量不变，只比较：
  - `SingleStreamTinyTCN`
  - `DualStreamTinyTCN`
- 先看目标难工况：
  - `工况1`
  - `工况3`
  - `工况17`
  - `工况18`
- 如果目标难工况上出现正信号，再自动补一组容易工况对照：
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

## 通道分组

- `strain`：
  - 所有 `应变传感器*.chdata`
- `acc`：
  - 所有 `WSMS*.Acc*`

## 双流结构

- `strain` 分支单独编码；
- `acc` 分支单独编码；
- 两路各自做 `TinyTCN` 编码与全局池化；
- 在 head 前拼接两路表示，再输出风速回归结果。

## 运行方式

```powershell
.venv\Scripts\python.exe src/try/027_dualstream_tinytcn_quickcheck/run_dualstream_tinytcn_quickcheck.py
```

## 输出

- 输出目录：`outputs/try/027_dualstream_tinytcn_quickcheck/`
- 固定产物：
  - `target_case_variant_comparison.csv`
  - `target_case_variant_summary.csv`
  - `control_case_variant_comparison.csv`
  - `control_case_variant_summary.csv`
  - `summary.md`
