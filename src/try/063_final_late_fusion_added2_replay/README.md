# 063 final 训练的 2s+8s 晚融合外推 added2

## 目标

- 检查仅在 `final` 带标签工况上训练的 `TinyTCN@2s`、`TinyTCN@8s` 以及 `2s+8s` 晚融合，是否可以外推到 `added2`。
- 若仓库中不存在可复用的 `full final deploy` 权重，则训练并持久化新的 deploy 模型。
- 在同一口径下同时回放 `added(21-24)` 与 `added2(25-30)`，方便比较外部域变化。

## 输入

- `data/final/dataset_manifest.csv`
- `data/final/datasets/工况1-20.csv`
- `data/added/dataset_manifest.csv`
- `data/added/standardized_datasets/工况21-24.csv`
- `data/added2/dataset_manifest.csv`
- `data/added2/standardized_datasets/工况25-30.csv`

## 运行

```bash
uv run python src/try/063_final_late_fusion_added2_replay/run_final_late_fusion_added2_replay.py
```

强制重训：

```bash
uv run python src/try/063_final_late_fusion_added2_replay/run_final_late_fusion_added2_replay.py --force-retrain
```

## 输出

- `outputs/try/063_final_late_fusion_added2_replay/models/checkpoints/final_deploy_2s.pt`
- `outputs/try/063_final_late_fusion_added2_replay/models/checkpoints/final_deploy_2s_norm.npz`
- `outputs/try/063_final_late_fusion_added2_replay/models/checkpoints/final_deploy_8s.pt`
- `outputs/try/063_final_late_fusion_added2_replay/models/checkpoints/final_deploy_8s_norm.npz`
- `outputs/try/063_final_late_fusion_added2_replay/case_level_predictions.csv`
- `outputs/try/063_final_late_fusion_added2_replay/summary_by_domain.csv`
- `outputs/try/063_final_late_fusion_added2_replay/summary.md`
