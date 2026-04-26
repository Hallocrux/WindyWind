# 062 added2 域诊断

## 目标

- 把 `data/added2/` 的 `6` 个新补充工况接成与现有主线一致的标准化资产。
- 在不改动默认训练主线的前提下，诊断 `added2` 相对 `final + added` 的域位置。
- 检查 `added2` 的 `wind_speed / rpm` 标签与解析 RPM 链是否自洽。

## 输入

- `data/final/dataset_manifest.csv`
- `data/final/datasets/工况1-20.csv`
- `data/added/dataset_manifest.csv`
- `data/added/standardized_datasets/工况21-24.csv`
- `data/added2/dataset_manifest.csv`
- `data/added2/standardized_datasets/工况25-30.csv`
- `outputs/try/030_case_mechanism_clustering/case_embedding.csv`

## 运行

```bash
uv run python src/try/062_added2_domain_diagnosis/run_added2_domain_diagnosis.py
```

## 输出

- `outputs/try/062_added2_domain_diagnosis/case_inventory.csv`
- `outputs/try/062_added2_domain_diagnosis/nearest_reference_cases.csv`
- `outputs/try/062_added2_domain_diagnosis/added2_domain_summary.csv`
- `outputs/try/062_added2_domain_diagnosis/added2_feature_outliers_vs_final.csv`
- `outputs/try/062_added2_domain_diagnosis/label_rpm_consistency.csv`
- `outputs/try/062_added2_domain_diagnosis/mechanism_projection.png`
- `outputs/try/062_added2_domain_diagnosis/summary.md`
