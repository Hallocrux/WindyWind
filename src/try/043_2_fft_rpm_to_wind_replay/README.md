# 043_2 FFT RPM -> Wind 结果回放验证

## 目标

- 不重跑任何 FFT / TinyTCN 主实验；
- 直接复用已有 case 级 RPM 预测结果，验证：
  - `fft_rpm -> wind`
  - 与 `043` 中已有 `pred_rpm TinyTCN -> wind` 链路相比，误差改善多少；
- 保持 `043` 中 `rpm_knn4 / rpm_linear / ridge_rpm_to_wind` 的映射口径不变。

## 输入与口径

- FFT RPM 结果：
  - `outputs/try/043_1_fft_rpm_algorithm_search/case_level_predictions.csv`
- TCN deployable 参考：
  - `outputs/try/043_pred_rpm_deployability_check/rpm_to_wind_summary.csv`
  - `outputs/try/043_pred_rpm_deployability_check/rpm_to_wind_case_level_predictions.csv`
- 标量映射口径：
  - 复用 `src/try/043_pred_rpm_deployability_check/run_pred_rpm_deployability_check.py`
- 标签数据：
  - `data/final/dataset_manifest.csv`
  - `data/final/datasets/工况*.csv`
  - `data/added/dataset_manifest.csv`
  - `data/added/standardized_datasets/工况21-24.csv`

## 运行方式

```powershell
uv run python src/try/043_2_fft_rpm_to_wind_replay/run_fft_rpm_to_wind_replay.py
```

## 输出

- 输出目录：`outputs/try/043_2_fft_rpm_to_wind_replay/`
- 固定产物：
  - `fft_rpm_to_wind_case_level_predictions.csv`
  - `fft_rpm_to_wind_summary.csv`
  - `fft_vs_tcn_summary.csv`
  - `summary.md`
