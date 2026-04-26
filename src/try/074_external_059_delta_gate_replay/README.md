# 074 external 059 delta gate replay

## 目标

- 把 `059` 中的 `delta-only + trigger/gate` 思路迁到 external-first 口径；
- 不再混入 `final`；
- 验证：
  - 在 `added -> added2` 外推下，保守 trigger / gate 是否优于 `rpm_knn4`；
  - 在 `added + added2` external `LOOCV` 下，`delta-only` correction 是否还能稳定工作。

## 输入

- `outputs/try/069_added2_embedding_pca_projection/embedding_case_table.csv`
- `data/added/`
- `data/added2/`

## 方法摘要

- 检索空间：
  - 统一高维 case embedding
- `base`：
  - `rpm_knn4`
- `candidate`：
  - `delta_only_prototype_ridge @ w=0.5`
- `route / gate`：
  - `soft_gate_hgb`
  - `binary_hgb_t0.65`
  - `bucket_hgb`
  - `two_stage_hgb_t0.65`
  - `trigger_rule_cv`

## 评估协议

- `external_loocv`
  - 在 `added + added2` 的 `10` 个带标签工况上做 external `LOOCV`
- `added_to_added2`
  - 只用 `added(21-24)` 训练
  - 只在 `added2(25-30)` 测试

## 运行方式

```powershell
uv run python src/try/074_external_059_delta_gate_replay/run_external_059_delta_gate_replay.py
```

## 输出

- 输出目录：`outputs/try/074_external_059_delta_gate_replay/`
- 主要文件：
  - `all_case_predictions.csv`
  - `summary_by_protocol.csv`
  - `summary_by_protocol_and_domain.csv`
  - `prototype_feature_table.csv`
  - `gate_feature_table.csv`
  - `gate_training_table.csv`
  - `neighbor_table.csv`
  - `summary.md`
