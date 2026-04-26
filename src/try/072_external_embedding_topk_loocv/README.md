# 072 external embedding top-k LOOCV

## 目标

- 以当前 `added + added2` 的 `10` 个带标签样本为外部域池；
- 基于 `069` 已导出的 `TinyTCN 2s+8s` 高维 case embedding；
- 做外部域内部 `LOOCV`；
- 在当前 `rpm_knn4 + embedding residual` 思路上显式加入 `top-k` 选择；
- 对比：
  - `rpm_knn4`
  - `embedding_knn4`
  - `rpm_knn4 + global embedding residual ridge`
  - `rpm_knn4 + top-k residual mean`
  - `rpm_knn4 + top-k residual ridge`

## 输入

- `outputs/try/069_added2_embedding_pca_projection/embedding_case_table.csv`

## 运行方式

```powershell
uv run python src/try/072_external_embedding_topk_loocv/run_external_embedding_topk_loocv.py
```

## 输出

- 输出目录：`outputs/try/072_external_embedding_topk_loocv/`
- 主要文件：
  - `case_level_predictions.csv`
  - `summary_by_variant.csv`
  - `summary_by_domain.csv`
  - `neighbor_table.csv`
  - `summary.md`
  - `plots/pred_vs_true.png`

## 说明

- 评估协议只做 `added + added2` 的 external `LOOCV`。
- residual 监督目标继续使用训练子集内部的近似 `OOF rpm_knn4 residual`。
