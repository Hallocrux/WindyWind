# 069 added2 embedding PCA projection

## 目标

- 参考 `057_embedding_space_diagnosis` 的统一 `TinyTCN 2s+8s` embedding 口径；
- 复用 `057` 已落盘的 `2s / 8s` encoder checkpoint，不重新训练；
- 把 `added2(工况25-30)` 纳入统一坐标系；
- 对 `final + added + added2` 做 PCA 二维投影；
- 输出一张突出 `added2` 位置的二维投影图像。

## 输入

- `outputs/try/057_embedding_space_diagnosis/models/checkpoints/unified_all_2s.pt`
- `outputs/try/057_embedding_space_diagnosis/models/checkpoints/unified_all_8s.pt`
- 对应 `norm.npz / json`
- `data/final/dataset_manifest.csv`
- `data/final/datasets/工况*.csv`
- `data/added/dataset_manifest.csv`
- `data/added/standardized_datasets/工况21-24.csv`
- `data/added2/dataset_manifest.csv`
- `data/added2/standardized_datasets/工况25-30.csv`

## 运行方式

```powershell
uv run python src/try/069_added2_embedding_pca_projection/run_added2_embedding_pca_projection.py
```

## 输出

- 输出目录：`outputs/try/069_added2_embedding_pca_projection/`
- 主要文件：
  - `embedding_case_table.csv`
  - `embedding_pca_coords.csv`
  - `added2_projection_summary.csv`
  - `summary.md`
  - `plots/pca_by_source_domain.png`
  - `plots/pca_added2_focus.png`
  - `plots/pca_added2_projection_panel.png`

## 说明

- `工况2` 只做前向投影，不参与任何训练。
- PCA 只用于可视化，不改变后续模型口径。
- `added2` 图中会被单独高亮，便于和 `final / added` 参考簇做直观比较。
