# 081 final 训练的 acc-only 2s+8s 晚融合外推 added2

## 目标

- 补测此前未单独验证的：
  - `TinyTCN@2s`
  - `TinyTCN@8s`
  - `TinyTCN 2s+8s late fusion`
- 但输入只保留 `acc_only`；
- 训练口径对齐 `063`：
  - 只在 `final` 带标签工况上训练；
  - 回放到 `added(21-24)` 与 `added2(25-30)`；
- 并与 `063` 的 `all_channels` 结果做同表对照。

## 输入

- `data/final/dataset_manifest.csv`
- `data/final/datasets/工况1-20.csv`
- `data/added/dataset_manifest.csv`
- `data/added/standardized_datasets/工况21-24.csv`
- `data/added2/dataset_manifest.csv`
- `data/added2/standardized_datasets/工况25-30.csv`
- `outputs/try/063_final_late_fusion_added2_replay/summary_by_domain.csv`
- `outputs/try/063_final_late_fusion_added2_replay/case_level_predictions.csv`

## 运行

```powershell
uv run python src/try/081_final_late_fusion_acc_only_added2_replay/run_final_late_fusion_acc_only_added2_replay.py
```

强制重训：

```powershell
uv run python src/try/081_final_late_fusion_acc_only_added2_replay/run_final_late_fusion_acc_only_added2_replay.py --force-retrain
```

## 输出

- `outputs/try/081_final_late_fusion_acc_only_added2_replay/models/checkpoints/final_deploy_acc_only_2s.pt`
- `outputs/try/081_final_late_fusion_acc_only_added2_replay/models/checkpoints/final_deploy_acc_only_8s.pt`
- `outputs/try/081_final_late_fusion_acc_only_added2_replay/case_level_predictions.csv`
- `outputs/try/081_final_late_fusion_acc_only_added2_replay/summary_by_domain.csv`
- `outputs/try/081_final_late_fusion_acc_only_added2_replay/summary_by_domain_with_063_reference.csv`
- `outputs/try/081_final_late_fusion_acc_only_added2_replay/case_compare_with_063_reference.csv`
- `outputs/try/081_final_late_fusion_acc_only_added2_replay/summary.md`
