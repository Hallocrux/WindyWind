# 057 embedding space diagnosis

## 目标

- 建立统一坐标系下的 `TinyTCN 2s+8s case embedding`；
- 诊断 `embedding_concat` 是否适合作为默认检索空间；
- 同时输出：
  - case embedding 表
  - PCA 可视化
  - pairwise 距离热图
  - `top-k` 邻居表
  - hubness 统计
  - PCA 平面上的邻居边图

## 核心问题

- `embedding_concat` 是否具有稳定的局部结构；
- `added` 工况在统一 embedding 空间中是“有桥接”还是“完全断裂”；
- 哪些工况在该空间中经常被选为邻居；
- 当前 `top-k` 检索结果是否符合直觉上的局部参考域。

## 输入

- `data/final/dataset_manifest.csv`
- `data/final/datasets/工况*.csv`
- `data/added/dataset_manifest.csv`
- `data/added/standardized_datasets/工况21-24.csv`
- 优先复用：
  - `outputs/try/053_support_window_residual_quickcheck/models/checkpoints/`

## 运行方式

```powershell
uv run python src/try/057_embedding_space_diagnosis/run_embedding_space_diagnosis.py
```

## 输出

- 输出目录：`outputs/try/057_embedding_space_diagnosis/`
- 主要文件：
  - `embedding_case_table.csv`
  - `embedding_pca_coords.csv`
  - `pairwise_distance_matrix.csv`
  - `knn_neighbors.csv`
  - `hubness_counts.csv`
  - `summary.md`
  - `plots/pca_by_domain.png`
  - `plots/pca_by_wind_speed.png`
  - `plots/pca_top1_edges.png`
  - `plots/pairwise_distance_heatmap.png`
  - `plots/hubness_bar.png`
