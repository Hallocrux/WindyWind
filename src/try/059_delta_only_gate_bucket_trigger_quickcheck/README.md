# 059 delta-only gate / bucket / trigger quickcheck

## 目标

- 延续 `058` 的结论：
  - `embedding_concat` 继续作为默认检索空间；
  - correction 先收缩到 `delta-only`；
- 在此基础上，只验证更保守的三类路由：
  - `soft gate`
  - `bucket`
  - `trigger`
- 关注的问题是：
  - 能不能尽量保住 added 上的局部收敛；
  - 同时把 `delta-only` 在 `final_focus` 上的额外伤害压低。

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

- 检索空间：
  - 继续使用 `TinyTCN 2s+8s case embedding concat`
- correction head：
  - 继续使用 `058` 的 `delta-only prototype ridge @ w=0.5`
- 路由层：
  - 先对 outer fold 的训练工况做 inner-LOO，得到 `delta-only` 候选的 OOF 预测；
  - 再基于 prototype 特征 + 机制特征，学习：
    - `soft gate`
    - `binary trigger`
    - `bucket`
    - `two-stage trigger+bucket`
  - 额外补一个小型 `trigger_rule_cv`，只做阈值搜索，不做连续回归。

## 输入与复用

- 主训练域：
  - `data/final/dataset_manifest.csv`
  - `data/final/datasets/工况*.csv`
- 补充域：
  - `data/added/dataset_manifest.csv`
  - `data/added/standardized_datasets/工况21-24.csv`
- encoder / 清洗 / 数据加载：
  - 复用 `src/try/058_prototype_head_ablation_quickcheck/`
  - 复用 `src/try/053_support_window_residual_quickcheck/`
  - 机制特征复用 `src/try/047_soft_gate_quickcheck/`

## 运行方式

```powershell
uv run python src/try/059_delta_only_gate_bucket_trigger_quickcheck/run_delta_only_gate_bucket_trigger_quickcheck.py
```

## 输出

- 输出目录：`outputs/try/059_delta_only_gate_bucket_trigger_quickcheck/`
- 主要文件：
  - `case_level_predictions.csv`
  - `summary_by_domain.csv`
  - `gate_training_table.csv`
  - `gate_feature_table.csv`
  - `summary.md`
