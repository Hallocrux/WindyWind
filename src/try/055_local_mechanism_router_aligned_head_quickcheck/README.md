# 055 local mechanism router aligned head quickcheck

## 目标

- 验证“先限制到局部相似机制区域，再在局部区域里做表示对齐，再由一个受约束小头输出最终风速”是否比直接做单维 residual 更合理；
- 继续复用代表性 holdout：
  - `工况1`
  - `工况3`
  - `工况17`
  - `工况18`
  - `工况21`
  - `工况22`
  - `工况23`
  - `工况24`
- 新增两条对照：
  - `局部机制池 + residual kNN`
  - `局部机制池 + aligned representation + bounded head`

## 复用策略

- `2s / 8s` encoder 优先复用：
  - `outputs/try/053_support_window_residual_quickcheck/models/checkpoints/`
- `gate / mechanism` 特征复用：
  - `outputs/try/047_soft_gate_quickcheck/gate_feature_table.csv`
- `052 / 053` 的旧对照直接复用已落盘预测；
- 只有在 `053` 对应 fold checkpoint 缺失时，才会补训 encoder，并把新 checkpoint 落到当前实验输出目录下。

## 方法摘要

- `router`
  - 先按 case-level mechanism feature 选出局部参考池；
- `local residual knn`
  - 只在局部机制池内做 `2s+8s` case embedding residual kNN；
- `aligned head`
  - 在局部机制池内按 embedding 做 support attention；
  - 构造目标 case 与 support centroid 的对齐特征；
  - 用 bounded ridge head 预测受约束 correction。

## 运行方式

```powershell
uv run python src/try/055_local_mechanism_router_aligned_head_quickcheck/run_local_mechanism_router_aligned_head_quickcheck.py
```

## 输出

- 输出目录：`outputs/try/055_local_mechanism_router_aligned_head_quickcheck/`
- 主要文件：
  - `case_level_predictions.csv`
  - `summary_by_domain.csv`
  - `router_case_neighbors.csv`
  - `aligned_feature_table.csv`
  - `models/`
  - `summary.md`
