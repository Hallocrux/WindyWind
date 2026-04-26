# 086 contrastive condition embedding

## 目标

- 不使用 `wind_speed` 作为 TinyTCN encoder 的训练监督；
- 在 `final + added` 的所有可用窗口上做基于 `case_id` 的对比预训练；
- 导出 `added / added2` 的 case embedding；
- 只在 `added` 上训练 `rpm_knn4 + embedding residual ridge`；
- 在 `added2` 上评估，和 `071` 的 `case_mae = 0.6161` 做对照。

## 方法口径

- 训练 encoder：
  - `final` 带标签工况 `1, 3-20`
  - `added` 带标签工况 `21-24`
- 导出 embedding：
  - `added` 工况 `21-24`
  - `added2` 工况 `25-30`
- 窗长：
  - `2s`
  - `8s`
- encoder：
  - TinyTCN blocks
  - Temporal Pyramid Pooling `[1, 2, 4]`
  - projection head 输出 `64` 维 normalized embedding
- 训练目标：
  - supervised contrastive loss
  - 正样本由同一 `case_id` 的窗口与同一窗口增强视图构成
  - 不直接使用风速标签

## 运行方式

```powershell
uv run python src/try/086_contrastive_condition_embedding/run_contrastive_condition_embedding.py
```

导出全部工况高维向量并做 PCA：

```powershell
uv run python src/try/086_contrastive_condition_embedding/project_all_cases_pca.py
```

## 输出

- 输出目录：`outputs/try/086_contrastive_condition_embedding/`
- 主要文件：
  - `embedding_case_table.csv`
  - `case_level_predictions.csv`
  - `summary_by_protocol.csv`
  - `knn_neighbors.csv`
  - `summary.md`
  - `models/checkpoints/contrastive_2s.pt`
  - `models/checkpoints/contrastive_8s.pt`
  - `models/checkpoints/contrastive_2s_norm.npz`
  - `models/checkpoints/contrastive_8s_norm.npz`
  - `models/checkpoints/contrastive_2s.json`
  - `models/checkpoints/contrastive_8s.json`
  - `all_case_embedding_table.csv`
  - `all_case_embedding_pca_coords.csv`
  - `all_case_embedding_pca_summary.md`
  - `plots/all_case_embedding_pca.png`

## 说明

- `added2` 不参与 residual ridge 训练；
- 默认脚本会把 `added2` 只用于 embedding 前向导出与最终评估；
- 该探索只判断 contrastive condition embedding 能否替换 `071` 中复用的风速监督 TinyTCN embedding。
