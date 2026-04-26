# 078 repo fft rpm blend quickcheck

## 目标

- 使用仓库内 `043_1_fft_rpm_algorithm_search` 的 FFT RPM 口径；
- 复现其 added 外部域最优变体：
  - `hybrid_peak_1x_whole_window8_gate150`
- 在当前 added-first 口径下 quickcheck：
  - `rpm_true_knn4`
  - `rpm_repo_fft_knn4`
  - `rpm_repo_mix05_knn4`

## 输入

- 代码：
  - `src/try/043_1_fft_rpm_algorithm_search/run_fft_rpm_algorithm_search.py`
- 数据：
  - `data/added/standardized_datasets/工况21-24.csv`
  - `data/added2/standardized_datasets/工况25-30.csv`
  - `data/added/dataset_manifest.csv`
  - `data/added2/dataset_manifest.csv`

## 评估口径

- `added_to_added2`
- `external_loocv`

## 输出

- 输出目录：`outputs/try/078_repo_fft_rpm_blend_quickcheck/`
- 主要文件：
  - `fft_feature_table.csv`
  - `case_level_predictions.csv`
  - `summary_by_protocol.csv`
  - `summary_by_protocol_and_domain.csv`
  - `summary.md`
