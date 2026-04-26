# 066 reuse embedding domain split

## 目标

- 复用 `2026-04-08` 已落盘的 `057` 统一 `TinyTCN 2s+8s` embedding checkpoint；
- 不再训练新的 encoder；
- 在 `final + added + added2` 的工况并集上导出统一坐标系 embedding；
- 基于 `embedding_concat` 做新的 case-level 科学域划分；
- 为后续 `067` 的 `Leave-One-Domain-Out` 训练生成稳定的 `domain_assignment.csv`。

## 输入

- 旧 embedding 资产：
  - `outputs/try/057_embedding_space_diagnosis/models/checkpoints/unified_all_2s.pt`
  - `outputs/try/057_embedding_space_diagnosis/models/checkpoints/unified_all_8s.pt`
  - 对应 `norm.npz / json`
- 数据：
  - `data/final/dataset_manifest.csv`
  - `data/final/datasets/工况*.csv`
  - `data/added/dataset_manifest.csv`
  - `data/added/standardized_datasets/工况21-24.csv`
  - `data/added2/dataset_manifest.csv`
  - `data/added2/standardized_datasets/工况25-30.csv`

## 运行方式

```powershell
uv run python src/try/066_reuse_embedding_domain_split/run_reuse_embedding_domain_split.py
```

如需快速 smoke test，可保留默认参数直接运行；本探索不包含重新训练步骤。

## 输出

- 输出目录：`outputs/try/066_reuse_embedding_domain_split/`
- 主要文件：
  - `embedding_case_table.csv`
  - `embedding_reuse_validation.csv`
  - `cluster_selection_report.csv`
  - `domain_assignment.csv`
  - `domain_summary.csv`
  - `knn_neighbors.csv`
  - `summary.md`
  - `plots/`

## 说明

- `工况2` 只做 embedding 前向投影与后验赋域，不参与聚类拟合。
- 聚类只比较 `K=4` 与 `K=5`，默认优先选择满足最小样本约束且 silhouette 更高的解。
