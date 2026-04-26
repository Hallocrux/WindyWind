# 088 071 residual regressor ablation

## 目标

固定 `071` 的口径，只替换 residual regressor：

- embedding 表：`outputs/try/069_added2_embedding_pca_projection/embedding_case_table.csv`
- 主干：`rpm_knn4`
- residual target：训练子集内部 `rpm_knn4` 近似 OOF residual
- 主协议：`added(21-24) -> added2(25-30)`

核心问题：

- `071 | rpm_knn4 + embedding residual ridge` 的瓶颈是不是 Ridge 回归器本身；
- 是否存在同一输入、同一 residual target 下更强的替代回归器。

## 运行方式

```powershell
uv run python src/try/088_071_residual_regressor_ablation/run_071_residual_regressor_ablation.py
```

## 输出

- 输出目录：`outputs/try/088_071_residual_regressor_ablation/`
- 主要文件：
  - `all_case_predictions.csv`
  - `summary_by_protocol.csv`
  - `added_to_added2_compare_vs_071.csv`
  - `summary.md`
