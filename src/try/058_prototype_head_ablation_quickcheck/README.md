# 058 prototype head ablation quickcheck

## 目标

- 在 `056` 已经确认“embedding top-k + local prototype` 可实现，但 full input head 在 added 上不够稳”的前提下；
- 只做两个最小版 ablation，验证 added 退化是否主要来自 head 自由度：
  - `057a` 风格：`delta-only ridge`
  - `057b` 风格：`low-rank delta ridge`

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
- `052 / 053 / 056` 的旧对照直接复用已落盘预测；
- 只有在统一 fold checkpoint 缺失时，才会补训 encoder 并落到当前实验输出目录下。

## 方法摘要

### 057a 风格：delta-only ridge

- 参考池：
  - 仍然使用 `embedding top-k`
- 小头输入只保留：
  - `delta`
  - `|delta|`
  - `base_pred`
  - `dist_stats`

### 057b 风格：low-rank delta ridge

- 参考池：
  - 仍然使用 `embedding top-k`
- 先对训练 fold 的 `delta` 做轻量 PCA；
- correction 只从低维 `delta_pca` 里输出；
- 不再把 `h`、`h_ref`、`|delta|`、`base_pred`、`dist_stats` 一起喂给小头。

## 运行方式

```powershell
uv run python src/try/058_prototype_head_ablation_quickcheck/run_prototype_head_ablation_quickcheck.py
```

## 输出

- 输出目录：`outputs/try/058_prototype_head_ablation_quickcheck/`
- 主要文件：
  - `case_level_predictions.csv`
  - `summary_by_domain.csv`
  - `reference_neighbors.csv`
  - `prototype_feature_table.csv`
  - `delta_pca_feature_table.csv`
  - `models/`
  - `summary.md`
