# 046 true rpm + acc residual quickcheck

## 目标

- 在 `true_rpm` 可用于推理的前提下，重做 `rpm-only` 上界；
- 用同一评估口径比较：
  - `linear`
  - `ridge`
  - `knn4`
  - `spline(GAM-like)`
- 再在每条 `rpm-only` 基线上叠加一个只使用加速度统计特征的残差修正分支；
- 快速判断：
  - `rpm` 是否已经足够强到可以做主干；
  - `acc-only residual` 能否稳定提供小幅正增益。

## 输入

- `data/final/dataset_manifest.csv`
- `data/final/datasets/工况{ID}.csv`
- `data/added/dataset_manifest.csv`
- `data/added/standardized_datasets/工况21-24.csv`

## 运行方式

```powershell
uv run python src/try/046_true_rpm_acc_residual_quickcheck/run_true_rpm_acc_residual_quickcheck.py
```

## 输出

- 输出目录：`outputs/try/046_true_rpm_acc_residual_quickcheck/`
- 主要文件：
  - `case_level_predictions.csv`
  - `summary_by_protocol.csv`
  - `summary_by_variant.csv`
  - `feature_table.csv`
  - `summary.md`

## 方法说明

- `rpm-only` 只使用工况级 `rpm -> wind` 映射；
- `acc residual` 只使用窗口级加速度统计特征，预测：
  - `residual = true_wind - rpm_only_pred`
- 最终预测：
  - `pred = rpm_only_pred + residual_pred`
- 同时保留一个保守残差版本：
  - `pred = rpm_only_pred + 0.4 * residual_pred`
- 评估协议同时输出：
  - `final_loco`
  - `added_external`
