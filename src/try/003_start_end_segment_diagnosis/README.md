# 003 开始/中段/结束 15s 差异诊断

## 目标

- 复用当前表格主线的清洗与切窗逻辑，检查每个工况在开始、中段、结束这三个 `15s` 时间段内的窗口表现是否存在系统性差异。
- 重点回答两个问题：
  - 开始段是否因为风机尚未进入稳态而显著不同；
  - 结束段是否也存在明显偏离，以至于应在训练时视为应排除的边界垃圾窗口。

## 输入与口径

- 数据源：`data/final/datasets/`
- 清洗逻辑：复用 `src/current/data_loading.py`
- 切窗逻辑：复用 `src/current/features.py`
- 当前窗口口径：
  - 采样率 `50Hz`
  - 窗长 `5s`（250 点）
  - 步长 `2.5s`（125 点）
- 本探索从每个工况中截取三个时间段：
  - `start`：开始 `15s`
  - `middle`：以全程中点为中心的 `15s`
  - `end`：最后 `15s`
- 只统计“完整落入该时间段”的窗口，避免跨段窗口把边界效应抹平。

## 输出

- 输出目录：`outputs/try/003_start_end_segment_diagnosis/`
- 主要产物：
  - `segment_window_summary.csv`
    - 每个工况、每个时间段的窗口数、缺失窗口占比、平均能量等摘要
  - `segment_distance_summary.csv`
    - 每个工况的 `start-middle`、`end-middle` 特征距离和相对段内离散度
  - `segment_overall_summary.csv`
    - 跨工况汇总后的整体结论指标
  - `segment_differences.png`
    - 边界段与中段差异的总览图
  - `summary.md`
    - 面向人工阅读的结论摘要

## 运行方式

```powershell
uv run python src/try/003_start_end_segment_diagnosis/analyze_segment_differences.py
```

也可以显式调整时间段长度和输出目录：

```powershell
uv run python src/try/003_start_end_segment_diagnosis/analyze_segment_differences.py --segment-seconds 15 --output-dir outputs/try/003_start_end_segment_diagnosis
```
