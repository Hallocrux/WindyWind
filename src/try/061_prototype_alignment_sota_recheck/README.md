# 061 prototype alignment SOTA recheck

## 目标

- 不再做新的 quickcheck 花样；
- 只复核 `060 embedding prototype alignment` 当前版，与两条既有目标线做同主题对照：
  - `final SOTA`
  - `added 上界 SOTA`

## 对照目标

- `final SOTA`
  - `TinyTCN_multiscale_late_fusion_2s_8s`
  - 口径：`full final LOCO`
- `added 上界 SOTA`
  - `rpm_knn4 + TinyTCN all_channels midband @ w=0.3`
  - 口径：`full_final_pool -> added 21-24`

## 当前版

- `060 | rpm_knn4 + embedding_prototype_alignment_ridge`
- `060 | rpm_knn4 + embedding_prototype_alignment_ridge @ w=0.5`

## 运行方式

```powershell
uv run python src/try/061_prototype_alignment_sota_recheck/run_prototype_alignment_sota_recheck.py
```

## 输出

- 输出目录：`outputs/try/061_prototype_alignment_sota_recheck/`
- 主要文件：
  - `final_case_level_predictions.csv`
  - `final_summary.csv`
  - `added_case_level_predictions.csv`
  - `added_summary.csv`
  - `comparison_to_sota.csv`
  - `prototype_neighbors.csv`
  - `alignment_feature_table.csv`
  - `summary.md`
