# 083 competition test embedding PCA projection

## 目标

- 复用 `057` 已落盘的 `TinyTCN 2s+8s` encoder checkpoint；
- 参考 `069` 的统一 PCA 口径；
- 将当前已跑过的四组竞赛测试输入一并提取高维 case embedding；
- 与 `final + added + added2` 一起做 PCA 二维投影；
- 输出一张“参考底图 + 测试样本高亮”的二维投影图。

## 当前四组测试输入

- `data/test/竞赛预测风速工况.csv`，`rpm=204`
- `data/test/竞赛预测风速工况2 .csv`，`rpm=192`
- `data/test/竞赛预测风速工况2 .csv`，`rpm=228`
- `data/test/竞赛预测风速工况4.csv`，`rpm=250`

## 运行方式

```powershell
uv run python src/try/083_competition_test_embedding_pca_projection/run_competition_test_embedding_pca_projection.py
```

## 输出

- 输出目录：`outputs/try/083_competition_test_embedding_pca_projection/`
- 主要文件：
  - `embedding_case_table.csv`
  - `embedding_pca_coords.csv`
  - `competition_test_projection_summary.csv`
  - `summary.md`
  - `plots/pca_with_competition_tests.png`
  - `plots/pca_competition_focus.png`
  - `plots/pca_competition_projection_panel.png`

## 说明

- PCA 仅用于可视化，不改变后续推理口径；
- `竞赛预测风速工况2 .csv` 的 `192 rpm` 与 `228 rpm` 共享同一份原始信号，因此 embedding 坐标预计重合；
- 这类重合不会被人为改动，只会在图上做标签区分。
