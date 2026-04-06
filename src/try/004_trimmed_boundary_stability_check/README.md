# 004 裁剪后边界稳定性复核

## 目标

- 按当前决定的口径，先删除每个工况开头和结尾的连续缺失段。
- 在裁剪后的数据上，继续检查新的开始段、中段、结束段是否还表现出明显的非稳态。
- 重点判断：
  - 裁剪后开头是否仍像静止后启动阶段；
  - 裁剪后开头是否存在明显外部激励导致的异常高波动或快速衰减；
  - 裁剪后结尾是否仍明显偏离中段稳态。

## 输入与口径

- 数据源：`data/final/datasets/`
- 清洗逻辑：复用 `src/current/data_loading.py`
- 裁剪规则：
  - 删除原始数据中首部连续缺失段；
  - 删除原始数据中尾部连续缺失段；
  - 中间部分保留，并继续使用当前主线的缺失填充方式。
- 比较口径：
  - 使用主线 `5s` 窗长、`2.5s` 步长；
  - 在裁剪后的数据上取 `start / middle / end` 三个 `15s` 时间段；
  - 只统计完整落入对应时间段的窗口。

## 输出

- 输出目录：`outputs/try/004_trimmed_boundary_stability_check/`
- 主要产物：
  - `trimmed_segment_summary.csv`
  - `trimmed_distance_summary.csv`
  - `trimmed_stationarity_summary.csv`
  - `trimmed_boundary_overview.png`
  - `summary.md`

## 运行方式

```powershell
uv run python src/try/004_trimmed_boundary_stability_check/check_trimmed_boundary_stability.py
```
