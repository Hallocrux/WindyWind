# 060 embedding top-k prototype alignment quickcheck

## 目标

- 明确按“新的融合”定义推进：
  - `embedding top-k` 只负责检索局部参考域；
  - 先把邻居 embedding 融成 `local prototype`；
  - 再在表征层比较 / 对齐 `target` 与 `prototype`；
  - 最后只做一个有界、保守的预测修正。
- 本实验显式避免把融合退化成：
  - 邻居 `wind` 平均；
  - 邻居 `residual` 平均。

## holdout 工况

- `工况1`
- `工况3`
- `工况17`
- `工况18`
- `工况21`
- `工况22`
- `工况23`
- `工况24`

## 方法摘要

- 检索：
  - 使用 `TinyTCN 2s+8s case embedding concat`
  - 在训练池里做 `top-k` 检索
- prototype：
  - 对 top-k embedding 做距离加权均值，得到 `h_ref`
  - 同时统计局部支持集的逐维离散度
- alignment：
  - 计算 `delta = h - h_ref`
  - 用局部离散度对 `delta` 做归一化
  - 对归一化后的偏移做 `tanh` 收缩，得到 `aligned target`
- bounded head：
  - 只使用：
    - `prototype` 的压缩坐标
    - `aligned target` 相对 `prototype` 的压缩位移
    - prototype 可信度 / 离散度统计
  - 输出 bounded correction

## 运行方式

```powershell
uv run python src/try/060_embedding_topk_prototype_alignment_quickcheck/run_embedding_topk_prototype_alignment_quickcheck.py
```

## 输出

- 输出目录：`outputs/try/060_embedding_topk_prototype_alignment_quickcheck/`
- 主要文件：
  - `case_level_predictions.csv`
  - `summary_by_domain.csv`
  - `prototype_neighbors.csv`
  - `alignment_feature_table.csv`
  - `summary.md`
