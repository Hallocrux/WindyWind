# 070 added added2 PCA line regression

## 目标

- 基于 `069` 已导出的 `TinyTCN 2s+8s` PCA 坐标；
- 只取 `added + added2` 的带标签工况；
- 把这批点视作一条潜在的一维流形，拟合 PCA 平面内的主轴线；
- 沿该主轴投影为一维坐标，再做风速回归 quickcheck；
- 对比：
  - 沿主轴的一维线性回归；
  - 沿主轴的一维二次回归；
  - 直接用二维 `pca1 + pca2` 做线性回归。

## 输入

- `outputs/try/069_added2_embedding_pca_projection/embedding_pca_coords.csv`

## 运行方式

```powershell
uv run python src/try/070_added_added2_pca_line_regression/run_added_added2_pca_line_regression.py
```

## 输出

- 输出目录：`outputs/try/070_added_added2_pca_line_regression/`
- 主要文件：
  - `added_added2_with_line_projection.csv`
  - `loocv_summary.csv`
  - `added_train_added2_test_summary.csv`
  - `summary.md`
  - `plots/pca_line_projection.png`
  - `plots/projection_vs_wind.png`

## 说明

- 本探索只验证“PCA 图上的那条线”是否真的承载稳定风速映射。
- 这是诊断性 quickcheck，不直接替代正式主线。
