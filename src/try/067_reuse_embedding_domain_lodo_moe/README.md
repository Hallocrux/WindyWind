# 067 reuse embedding domain LODO MoE

## 目标

- 读取 `066` 生成的 `domain_assignment.csv`；
- 不再沿用 `final_loco + added_external` 的固定协议；
- 复用 `057` 的统一 `TinyTCN 2s+8s` encoder checkpoint；
- 在新的 learned domains 上做 `Leave-One-Domain-Out`；
- 精简对照只跑：
  - `A0_rpm_knn4`
  - `A3_sparse_router_moe`

## 输入

- 域划分：
  - `outputs/try/066_reuse_embedding_domain_split/domain_assignment.csv`
- embedding checkpoint：
  - `outputs/try/057_embedding_space_diagnosis/models/checkpoints/unified_all_2s.pt`
  - `outputs/try/057_embedding_space_diagnosis/models/checkpoints/unified_all_8s.pt`
  - 对应 `norm.npz / json`
- 数据：
  - `data/final/`
  - `data/added/`
  - `data/added2/`

## 运行方式

```powershell
uv run python src/try/067_reuse_embedding_domain_lodo_moe/run_reuse_embedding_domain_lodo_moe.py
```

快速 smoke test：

```powershell
uv run python src/try/067_reuse_embedding_domain_lodo_moe/run_reuse_embedding_domain_lodo_moe.py --limit-holdout-domains 2 --freeze-epochs 1 --finetune-epochs 1
```

## 输出

- 输出目录：`outputs/try/067_reuse_embedding_domain_lodo_moe/`
- 主要文件：
  - `case_level_predictions.csv`
  - `summary_by_learned_domain.csv`
  - `summary_by_raw_source.csv`
  - `router_activation_table.csv`
  - `prototype_retrieval_stats.csv`
  - `training_log.csv`
  - `fold_metadata.csv`
  - `static_checks.csv`
  - `summary.md`

## Kaggle 发布

```powershell
uv run python src/try/067_reuse_embedding_domain_lodo_moe/publish_to_kaggle.py --skip-remote
```

该脚本会准备：

- `code dataset`
- `data dataset`
- `artifact dataset`
- `kernel`

默认 kernel 在 Kaggle 上只执行 `067`，不会重新聚类、不会重新训练 unified embedding。
