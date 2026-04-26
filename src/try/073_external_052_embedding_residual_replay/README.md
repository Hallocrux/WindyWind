# 073 external 052 embedding residual replay

## 目标

- 把 `052` 中最稳的旧方案：
  - `rpm_knn4 + embedding_residual_knn4_concat_2s_8s @ w=0.5`
  放到新的 external-first 口径下重新评估；
- 不再混入 `final`；
- 直接回答：
  - 在 `added -> added2` 外推下，它是否优于 `rpm_knn4`；
  - 在 `added + added2` external `LOOCV` 下，它是否优于 `rpm_knn4`。

## 输入

- `outputs/try/069_added2_embedding_pca_projection/embedding_case_table.csv`

## 方法摘要

- 外部域样本池只包含：
  - `added(工况21-24)`
  - `added2(工况25-30)`
- `base`：
  - `rpm_knn4`
- `052 replay`：
  - 用统一高维 case embedding 做 `k=4` 邻居检索；
  - 训练集内部先构造 `rpm_knn4` 的近似 `OOF residual`；
  - 对 holdout case 的 residual 做距离加权平均；
  - 最终预测：
    - `pred = rpm_knn4 + 0.5 * residual_knn4`

## 评估协议

- `external_loocv`
  - 在 `added + added2` 的 `10` 个带标签工况上做 external `LOOCV`
- `added_to_added2`
  - 只用 `added(21-24)` 训练
  - 只在 `added2(25-30)` 测试

## 运行方式

```powershell
uv run python src/try/073_external_052_embedding_residual_replay/run_external_052_embedding_residual_replay.py
```

## 输出

- 输出目录：`outputs/try/073_external_052_embedding_residual_replay/`
- 主要文件：
  - `all_case_predictions.csv`
  - `summary_by_protocol.csv`
  - `summary_by_protocol_and_domain.csv`
  - `neighbor_table.csv`
  - `summary.md`
