# 065 sparse router residual MoE V1

## 目标

- 在 `2026-04-08` 已有 `gate / embedding residual / prototype` 证据基础上；
- 用一个更小、更统一的训练版验证：
  - `rpm-first + shared encoder + sparse router + bounded residual experts`
  - 是否值得继续；
- `V1` 不追当前全仓库 `SOTA`，而是优先回答：
  - unified sparse residual-MoE 是否成立；
  - 路由行为是否可解释；
  - `L_noharm` 是否真的能保住 `base`。

## V1 固定约束

- `base = rpm_knn4`
- shared encoder 复用现有 `TinyTCN 2s+8s` encoder 权重初始化
- `3 experts`
  - `Expert0 = no-op`
  - `Expert1 = global residual expert`
  - `Expert2 = prototype delta-only residual expert`
- router 保留一次性 sparse `top-k` 激活
- 所有 expert 只输出 `bounded residual`
- 保留 `L_noharm`

## 输入与复用

- 主训练域：
  - `data/final/dataset_manifest.csv`
  - `data/final/datasets/工况*.csv`
- 补充域：
  - `data/added/dataset_manifest.csv`
  - `data/added/standardized_datasets/工况21-24.csv`
- encoder / embedding 口径优先复用：
  - `src/try/051_tcn_embedding_knn_residual/`
  - `src/try/052_tcn_embedding_window_signal_quickcheck/`
  - `src/try/053_support_window_residual_quickcheck/`
- prototype / delta-only 约束优先复用：
  - `src/try/056_embedding_topk_local_prototype_fusion/`
  - `src/try/058_prototype_head_ablation_quickcheck/`
  - `src/try/059_delta_only_gate_bucket_trigger_quickcheck/`

## 方法摘要

- `base`
  - 固定为 `rpm_knn4`
- `shared encoder`
  - `2s + 8s` case embedding concat
- `router`
  - 直接输出 `3` 个 expert logits
  - 使用 sparse `top-k`
- `Expert1`
  - 根据 `h + base_pred + rpm` 输出小幅 global residual
- `Expert2`
  - 先做 prototype retrieval
  - 只根据 `delta / |delta| / distance stats / base_pred` 输出小幅 residual
- `loss`
  - `Huber`
  - `L_noharm`
  - residual 幅度约束
- `evaluation`
  - `final LOCO`
  - `added external`
  - 最小 ablation：`A0-A4`

## 运行方式

```powershell
uv run python src/try/065_sparse_router_residual_moe_v1/run_sparse_router_residual_moe_v1.py
```

如需做缩小版 smoke test，可用：

```powershell
uv run python src/try/065_sparse_router_residual_moe_v1/run_sparse_router_residual_moe_v1.py --limit-final-holdouts 2 --freeze-epochs 1 --finetune-epochs 1
```

## 输出

- 输出目录：`outputs/try/065_sparse_router_residual_moe_v1/`
- 主要文件：
  - `case_level_predictions.csv`
  - `summary_by_domain.csv`
  - `router_activation_table.csv`
  - `expert_residual_stats.csv`
  - `expert_residual_summary.csv`
  - `prototype_retrieval_stats.csv`
  - `training_log.csv`
  - `fold_metadata.csv`
  - `static_checks.csv`
  - `summary.md`

## 设计稿

- 详细设计见：
  - `Docs/sparse_router_residual_moe_v1_design_2026-04-08.md`
