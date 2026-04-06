# 023 工况5 51帧锚点 5s 窗预测

## 目标

- 基于 `2026-04-06` 已确认的 `工况5` 视频手工标注片段：
  - selector 帧范围 `1975-2025`
  - 总计 `51` 帧
  - 时间跨度约 `0.8390s`
- 不重训短窗模型，继续使用当前已有的 `5s` TinyTCN 工况5 holdout 预测结果；
- 以这 `51` 帧片段的中心时刻为锚点，在显式假设下对齐到工况5表格时间轴，并选出最接近的 `5s` 窗做定向解释。

## 对齐假设

- `2026-04-06` 仓库内尚无“视频绝对时钟 -> 表格绝对时钟”的硬同步文件；
- 因此本 try 采用一个显式近似：
  - 用手工片段中心时刻在整段视频中的相对位置；
  - 映射到 `工况5.csv` 的整段有效时长；
  - 再在已有 `5s` 滑窗预测结果里选择中心时刻最近的窗口。

## 输入

- `outputs/annotations/test/summary.json`
- `data/video/VID_20260330_162635.mp4`
- `outputs/dataset_inventory.csv`
- `outputs/try/022_case5_video_manual_label_validation/case5_loco_window_predictions.csv`

## 运行方式

```powershell
uv run python src/try/023_case5_51frame_anchor_5s_window_prediction/run_case5_51frame_anchor_5s_window_prediction.py
```

## 输出

- 输出目录：`outputs/try/023_case5_51frame_anchor_5s_window_prediction/`
- 固定产物：
  - `alignment_summary.csv`
  - `aligned_window_prediction.csv`
  - `summary.md`

## 当前用途

- 这个 try 不是在声称“5s 模型直接预测了 51 帧”；
- 它是在当前缺少硬同步文件时，给出一个最明确、可复现、带假设的 `5s` 窗对齐解释。
