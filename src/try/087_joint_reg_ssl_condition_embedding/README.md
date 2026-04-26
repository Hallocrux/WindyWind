# 087 joint reg ssl condition embedding

## 目标

验证联合训练口径：

```text
L = L_reg(wind_speed) + lambda_ssl * L_ssl(case structure)
```

其中：

- `L_reg`：窗口级风速回归 MSE；
- `L_ssl`：基于 `case_id` 的 supervised contrastive loss，让同一工况窗口接近、不同工况窗口远离；
- 下游仍沿用 `071 / 086` 的 `added -> added2` 口径：
  - 在 `added(21-24)` 上训练 residual ridge；
  - 在 `added2(25-30)` 上评估。

## 方法口径

- encoder 训练数据：
  - `final` 带标签工况 `1, 3-20`
  - `added` 带标签工况 `21-24`
- 导出 embedding：
  - `added` 工况 `21-24`
  - `added2` 工况 `25-30`
- 窗长：
  - `2s`
  - `8s`
- 模型：
  - TinyTCN blocks
  - Temporal Pyramid Pooling `[1, 2, 4]`
  - regression head：预测窗口风速
  - projection head：输出 `64` 维 normalized embedding

## 运行方式

```powershell
uv run python src/try/087_joint_reg_ssl_condition_embedding/run_joint_reg_ssl_condition_embedding.py --lambda-ssl 0.1 --max-epochs 15 --force-retrain
```

## 输出

- 输出目录：`outputs/try/087_joint_reg_ssl_condition_embedding/`
- 主要文件：
  - `embedding_case_table.csv`
  - `case_level_predictions.csv`
  - `summary_by_protocol.csv`
  - `knn_neighbors.csv`
  - `embedding_pca_coords.csv`
  - `summary.md`
  - `models/checkpoints/joint_2s.pt`
  - `models/checkpoints/joint_2s_norm.npz`
  - `models/checkpoints/joint_2s.json`
  - `models/checkpoints/joint_8s.pt`
  - `models/checkpoints/joint_8s_norm.npz`
  - `models/checkpoints/joint_8s.json`
