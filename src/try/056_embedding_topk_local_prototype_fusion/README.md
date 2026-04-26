# 056 embedding top-k local prototype fusion

## 目标

- 验证是否可以把融合前移到高维表征层，而不是继续在单维 `wind residual` 上直接做邻居平均；
- 参考工况池只由 `TinyTCN 2s+8s case embedding` 的 `top-k` 决定；
- 在局部参考池上构造 `embedding prototype`，再让一个受约束小头根据：
  - 当前样本 embedding
  - 局部 reference prototype
  - 二者差异向量
  - `base_pred`
  输出一个小幅 correction。

## holdout 工况

- `工况1`
- `工况3`
- `工况17`
- `工况18`
- `工况21`
- `工况22`
- `工况23`
- `工况24`

## 复用策略

- `2s / 8s` encoder 优先复用：
  - `outputs/try/053_support_window_residual_quickcheck/models/checkpoints/`
- `052 / 053` 的旧对照直接复用已落盘预测；
- 只有在 `053` 对应 fold checkpoint 缺失时，才会补训 encoder，并把新 checkpoint 落到当前实验输出目录下。

## 方法摘要

- `reference pool`
  - 用 `2s+8s` concat 后的 case embedding 做 `top-k` 检索；
- `prototype`
  - 对局部参考池做距离加权均值，得到 `h_ref`；
- `fusion head`
  - 输入：
    - `h`
    - `h_ref`
    - `h - h_ref`
    - `|h - h_ref|`
    - `base_pred`
    - `top1 / topk` 距离统计
  - 头部：
    - `StandardScaler + RidgeCV`
    - 输出 bounded correction
    - 同时保留 `w=1.0` 与 `w=0.5` 两个版本。

## 运行方式

```powershell
uv run python src/try/056_embedding_topk_local_prototype_fusion/run_embedding_topk_local_prototype_fusion.py
```

## 输出

- 输出目录：`outputs/try/056_embedding_topk_local_prototype_fusion/`
- 主要文件：
  - `case_level_predictions.csv`
  - `summary_by_domain.csv`
  - `reference_neighbors.csv`
  - `prototype_feature_table.csv`
  - `models/`
  - `summary.md`
