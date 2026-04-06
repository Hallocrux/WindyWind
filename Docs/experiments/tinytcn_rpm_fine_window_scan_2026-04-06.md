# TinyTCN rpm 细窗长扫描（2026-04-06）

- 状态：`current`
- 首次确认：`2026-04-06`
- 最近复核：`2026-04-06`
- 数据范围：manifest 迁移后的 `工况1` 到 `工况20`
- 代码口径：
  - `src/try/024_tinytcn_rpm_fine_window_scan/`
- 证据入口：
  - `outputs/try/024_tinytcn_rpm_fine_window_scan/rpm_fine_window_scan_summary.csv`
  - `outputs/try/024_tinytcn_rpm_fine_window_scan/rpm_fine_window_scan_case_level_predictions.csv`
  - `outputs/try/024_tinytcn_rpm_fine_window_scan/summary.md`

## 1. 目标

在 `2026-04-06` 当前 TinyTCN rpm 回归口径下，继续细化 `2s` 到 `5s` 之间的窗长选择，回答：

- 更小时间窗口是否会更好；
- 还是 `2s-5s` 之间存在更优的中间值。

## 2. 方法口径

- 清洗逻辑：复用 `src/current.data_loading`
- 原始窗口构造：复用 `src/try/012_phase3_end_to_end_shortlist/phase3_end_to_end_lib.py`
- 模型：`TinyTCN`
- 监督目标：`dataset_manifest.csv` 中的 `rpm`
- 评估方式：
  - 分组单位为工况
  - `Leave-One-Condition-Out`
- 当前扫描窗长：
  - `2.0s`
  - `2.5s`
  - `3.0s`
  - `3.5s`
  - `4.0s`
  - `4.5s`
  - `5.0s`

## 3. 当前结果

### 3.1 [2026-04-06] 当前细窗长 full 最优点落在 `3.0s`

- 状态：`current`
- 首次确认：`2026-04-06`
- 最近复核：`2026-04-06`

全量 `LOCO` 结果：

- `TinyTCN @ 3.0s`
  - `case_mae = 5.1863`
  - `case_rmse = 8.5262`
  - `case_mape = 2.9500%`

对比其余候选：

- `TinyTCN @ 4.5s`
  - `case_mae = 7.7412`
- `TinyTCN @ 2.0s`
  - `case_mae = 7.8917`
- `TinyTCN @ 2.5s`
  - `case_mae = 8.0277`
- `TinyTCN @ 5.0s`
  - `case_mae = 8.3832`
- `TinyTCN @ 3.5s`
  - `case_mae = 9.0203`
- `TinyTCN @ 4.0s`
  - `case_mae = 9.2250`

### 3.2 [2026-04-06] 当前证据不支持“越短越好”，更支持存在中间最优窗长

- 状态：`current`
- 首次确认：`2026-04-06`
- 最近复核：`2026-04-06`

当前 full 扫描显示：

- `2.0s` 虽然优于 `5.0s`，但不是最优；
- `2.5s` 也不是最优；
- `3.0s` 明显优于当前扫描范围内的其他窗长；
- 这更像是“存在任务相关的中间最优窗长”，而不是“窗口越短越好”。

## 4. 当前判断

`2026-04-06` 的 rpm 细窗长扫描支持以下判断：

- 如果继续沿用当前 TinyTCN rpm 主线，默认候选窗长不应只在 `2.0s` 与 `5.0s` 之间二选一；
- 在当前数据口径和训练配置下，`3.0s` 是更值得优先复核和继续使用的候选；
- 后续如果继续围绕视频片段或更短局部片段做解释，应优先考虑 `2s-3s` 这一段，而不是直接回到 `5s` 长窗。
