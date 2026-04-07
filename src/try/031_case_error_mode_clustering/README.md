# 031 工况误差模式聚类探索

## 目标

- 在不重训模型的前提下，验证“高误差工况并不是同一种出错机制”；
- 构建 `per-case error profile table`；
- 做低维投影与聚类；
- 把第一层机制簇结果一起叠上去。

## 输入与口径

- 风速窗长误差：
  - `outputs/try/014_phase3_tcn_window_length_scan/tcn_window_scan_case_level_predictions.csv`
- RPM 细窗长误差：
  - `outputs/try/024_tinytcn_rpm_fine_window_scan/rpm_fine_window_scan_case_level_predictions.csv`
- 边界差异特征：
  - `outputs/try/003_start_end_segment_diagnosis/segment_distance_summary.csv`
- 第一层机制簇：
  - `outputs/try/030_case_mechanism_clustering/case_embedding.csv`

## 误差画像特征

- 风速任务：
  - `2s / 4s / 5s / 8s` 误差
  - 最优误差
  - `long-short delta`
- RPM 任务：
  - `2.0s - 5.0s` 各窗长误差
  - 最优误差
  - `5s - 3s delta`
- 边界段：
  - `start_middle_vs_within`
  - `end_middle_vs_within`
  - `start_end_vs_within`

## 运行方式

```powershell
.venv\Scripts\python.exe src/try/031_case_error_mode_clustering/run_case_error_mode_clustering.py
```

## 输出

- 输出目录：`outputs/try/031_case_error_mode_clustering/`
- 固定产物：
  - `case_error_profile_table.csv`
  - `case_error_embedding.csv`
  - `cluster_summary.csv`
  - `error_mode_pca_scatter.png`
  - `error_mode_heatmap.png`
  - `summary.md`
