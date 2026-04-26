# 071 external embedding regression quickcheck

## 目标

- 复用 `069` 已导出的 `TinyTCN 2s+8s` 高维 case embedding；
- 不做 PCA 压缩；
- 只在 `added + added2` 带标签工况上比较三条路线：
  - `embedding -> Ridge`
  - `embedding -> kNN`
  - `rpm_knn4 + embedding residual -> Ridge`
- 同时输出：
  - `added + added2` 的外部域 `LOOCV`
  - `added -> added2` 的外推结果

## 输入

- `outputs/try/069_added2_embedding_pca_projection/embedding_case_table.csv`

## 运行方式

```powershell
uv run python src/try/071_external_embedding_regression_quickcheck/run_external_embedding_regression_quickcheck.py
```

## 输出

- 输出目录：`outputs/try/071_external_embedding_regression_quickcheck/`
- 主要文件：
  - `all_case_predictions.csv`
  - `summary_by_protocol.csv`
  - `summary_by_protocol_and_domain.csv`
  - `summary.md`
  - `plots/pred_vs_true_external_loocv.png`
  - `plots/pred_vs_true_added_to_added2.png`

## 说明

- `rpm_knn4` 会作为同协议下的解析基线一起输出，方便判断 embedding 是否真的有增益。
- residual 目标使用训练子集内部的 `rpm_knn4` 近似 `OOF residual`，避免直接把 in-fold base 误差当监督。
