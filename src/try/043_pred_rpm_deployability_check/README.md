# 043 `pred_rpm` 可部署性验证

## 目标

- 把 `true_rpm -> rpm_to_wind` 的上界参考，改写成可部署的 `sensor -> pred_rpm -> rpm_to_wind` 链路；
- 验证 `pred_rpm` 替代 `true_rpm` 后，在 `final` 与 `added` 双域上的退化幅度；
- 判断哪种 `rpm -> wind` 标量映射更适合接收带误差的 `pred_rpm`。

## 输入与口径

- 主数据：
  - `data/final/dataset_manifest.csv`
  - `data/final/datasets/工况*.csv`
- added 数据：
  - `data/added/dataset_manifest.csv`
  - `data/added/standardized_datasets/工况21-24.csv`
- 清洗逻辑：复用 `src/current.data_loading`
- 原始窗口构造：复用 `src/try/012_phase3_end_to_end_shortlist/phase3_end_to_end_lib.py`
- rpm 预测主干：复用 `src/try/019_tinytcn_rpm_regression/tinytcn_rpm_lib.py`

## 默认变体

- `true_rpm -> rpm_knn4`
- `true_rpm -> rpm_linear`
- `true_rpm -> ridge_rpm_to_wind`
- `pred_rpm_3.0s -> rpm_knn4`
- `pred_rpm_3.0s -> rpm_linear`
- `pred_rpm_3.0s -> ridge_rpm_to_wind`
- `pred_rpm_2.0s -> rpm_linear`
- `pred_rpm_5.0s -> rpm_linear`

说明：

- `3.0s` 是 `2026-04-06` rpm 细窗长扫描的默认优先部署候选；
- `rpm_linear` 使用一维线性回归；
- `ridge_rpm_to_wind` 当前实现为 `rpm + rpm^2` 的二次 `Ridge` 平滑映射。

## 运行方式

```powershell
uv run python src/try/043_pred_rpm_deployability_check/run_pred_rpm_deployability_check.py
```

如需只跑主候选窗长：

```powershell
uv run python src/try/043_pred_rpm_deployability_check/run_pred_rpm_deployability_check.py --window-labels 3.0s
```

## 输出

- 输出目录：`outputs/try/043_pred_rpm_deployability_check/`
- 固定产物：
  - `rpm_case_level_predictions.csv`
  - `rpm_to_wind_case_level_predictions.csv`
  - `rpm_to_wind_summary.csv`
  - `deployable_vs_true_rpm_gap.csv`
  - `summary.md`
