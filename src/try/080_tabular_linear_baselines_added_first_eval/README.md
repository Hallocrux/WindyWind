# 080 tabular linear baselines added-first eval

## 目标

- 补测 added-first 子项目里此前没有单独进入同协议对照的线性表格基线；
- 在相同 external-first 口径下比较：
  - `TabularReference_G6_Ridge`
  - `Ridge + VIB_FT_RPM`
- 同时把以下结果放进同一汇总里作为参照：
  - `rpm_knn4`
  - `071 | rpm_knn4 + embedding residual ridge`

## 评估口径

- 主口径：`added_to_added2`
- 次口径：`external_loocv`

## 输入

- `data/added/standardized_datasets/工况21-24.csv`
- `data/added2/standardized_datasets/工况25-30.csv`
- `data/added/dataset_manifest.csv`
- `data/added2/dataset_manifest.csv`
- `outputs/try/071_external_embedding_regression_quickcheck/all_case_predictions.csv`

## 运行方式

```powershell
uv run python src/try/080_tabular_linear_baselines_added_first_eval/run_tabular_linear_baselines_added_first_eval.py
```

## 输出

- 输出目录：`outputs/try/080_tabular_linear_baselines_added_first_eval/`
- 主要文件：
  - `g6_feature_frame.csv`
  - `vib_ft_feature_frame.csv`
  - `all_case_predictions.csv`
  - `summary_by_protocol.csv`
  - `summary_by_protocol_and_domain.csv`
  - `summary.md`
