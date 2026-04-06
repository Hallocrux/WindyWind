# 025 TinyTCN 边界窗口误差检查

## 目标

- 在一个新的 try 中，复用 `2026-04-05` 已实现的 `TinyTCN` 风速回归口径；
- 专门检查 `工况1 / 3 / 17 / 18` 在 `LOCO` 评估下，不同窗口位置的预测误差是否显著不同；
- 把“边界段可能有害”从工作假设推进到更可检验的证据。

## 输入与口径

- 数据入口：
  - `data/final/dataset_manifest.csv`
  - `data/final/datasets/工况*.csv`
- 清洗逻辑：复用 `src/current.data_loading`
- 原始窗口构造：复用 `src/try/012_phase3_end_to_end_shortlist/`
- TinyTCN 风速回归与 `LOCO` 评估：复用 `src/try/013_phase3_cnn_tcn_smoke/`
- 默认窗口参数：
  - 采样率 `50Hz`
  - 窗长 `5s`
  - 步长 `2.5s`
- 目标工况：
  - `工况1`
  - `工况3`
  - `工况17`
  - `工况18`

## 位置分段规则

- 为了与 `003_start_end_segment_diagnosis` 保持口径一致：
  - `start`：开始 `15s`
  - `middle`：中间 `15s`
  - `end`：结束 `15s`
- 只有完整落在对应分段内的窗口才参与该分段比较。

## 统计检查

- 每个目标工况都输出：
  - 窗口级预测与绝对误差
  - `start / middle / end` 的误差汇总
  - `start vs middle`
  - `end vs middle`
  - `boundary(start+end) vs middle`
- 显著性检验默认使用置换检验，比较的是分段间平均绝对误差差值。

## 运行方式

```powershell
uv run python src/try/025_tinytcn_boundary_error_check/run_tinytcn_boundary_error_check.py
```

## 输出

- 输出目录：`outputs/try/025_tinytcn_boundary_error_check/`
- 固定产物：
  - `target_case_window_level_predictions.csv`
  - `target_case_segment_error_summary.csv`
  - `target_case_segment_tests.csv`
  - `target_case_error_over_time.png`
  - `target_case_error_boxplot.png`
  - `summary.md`
