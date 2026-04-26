# 064 added SOTA 回放 added2

## 目标

复核截至 2026-04-08 仓库中已有的两个相关 SOTA 口径在 `added2` 上的外推表现：

- `final SOTA`：`TinyTCN 2s+8s late fusion`，复用 `063` 已持久化的 full final deploy 权重；
- `added 上界 SOTA`：`rpm_knn4 + TinyTCN all_channels midband(3.0-6.0Hz) @ w=0.3`，复用 `042` 的训练口径与 10 个 seed 列表，对 `added2` 补充回放。

为同时评估 `added2`，本探索的信号列取 `final / added / added2` 的共同列；模型结构、频带、窗长和 seed 列表复用 `042`。

## 输入数据

- 训练域：`data/final/` 中带标签工况；
- 参考外部域：`data/added/` 的 `工况21-24`；
- 新外部域：`data/added2/` 的 `工况25-30`。

## 运行方式

```powershell
uv run python src/try/064_added_sota_added2_replay/run_added_sota_added2_replay.py
```

## 输出位置

输出目录：`outputs/try/064_added_sota_added2_replay/`

- `seed_case_level_predictions.csv`：midband 分支每个 seed 的 case 级预测；
- `seed_summary_by_domain.csv`：midband 分支每个 seed、每个外部域的汇总；
- `stability_overview_by_domain.csv`：midband 分支跨 seed 稳定性汇总；
- `direct_2s_8s_summary_by_domain.csv`：复用 `063` 的 direct `2s/8s/2s+8s` 汇总；
- `comparison_overview.csv`：midband SOTA 与 direct `2s+8s` 的同表对照；
- `added2_case_comparison.csv`：`added2` 每工况重点预测对照；
- `summary.md`：文字摘要。

## 模型资产

`TinyTCN 2s+8s` 资产复用 `outputs/try/063_final_late_fusion_added2_replay/models/checkpoints/`。

本探索会为 midband 分支写入新的 deploy checkpoint，路径为：

- `outputs/try/064_added_sota_added2_replay/models/checkpoints/final_deploy_midband_seed*_w5s.pt`
- `outputs/try/064_added_sota_added2_replay/models/checkpoints/final_deploy_midband_seed*_w5s_norm.npz`
- `outputs/try/064_added_sota_added2_replay/models/checkpoints/final_deploy_midband_seed*_w5s.json`
