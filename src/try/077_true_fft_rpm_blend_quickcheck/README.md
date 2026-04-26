# 077 true fft rpm blend quickcheck

## 目标

- 快速验证：
  - `rpm_mix = 0.5 * true_rpm + 0.5 * fft_rpm`
  是否在当前 added-first 口径下，对 `rpm` 主干显示正信号；
- 本探索只看 `rpm base`；
- 不引入 `embedding residual`，避免混淆信号来源。

## 输入

- FFT RPM 来源：
  - `src/from_others/1/predict_2.py`
- 数据：
  - `data/added/standardized_datasets/工况21-24.csv`
  - `data/added2/standardized_datasets/工况25-30.csv`
  - `data/added/dataset_manifest.csv`
  - `data/added2/dataset_manifest.csv`

## 评估口径

- `added_to_added2`
  - 训练：`added(21-24)`
  - 测试：`added2(25-30)`
- `external_loocv`
  - `added + added2` 的 `10` 个带标签工况 external `LOOCV`

## 比较对象

- `rpm_true_knn4`
- `rpm_fft_knn4`
- `rpm_mix05_knn4`

## 运行方式

```powershell
uv run python src/try/077_true_fft_rpm_blend_quickcheck/run_true_fft_rpm_blend_quickcheck.py
```

## 输出

- 输出目录：`outputs/try/077_true_fft_rpm_blend_quickcheck/`
- 主要文件：
  - `case_level_predictions.csv`
  - `fft_feature_table.csv`
  - `summary_by_protocol.csv`
  - `summary_by_protocol_and_domain.csv`
  - `summary.md`
