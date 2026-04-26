# added SOTA 回放 added2（2026-04-08）

- 状态：`current`
- 首次确认：`2026-04-08`
- 最近复核：`2026-04-08`
- 数据范围：
  - 训练域：`final` 带标签工况 `1,3-20`
  - 旧外部域：`added` 的 `工况21-24`
  - 新外部域：`added2` 的 `工况25-30`
- 代码口径：
  - `src/try/064_added_sota_added2_replay/`
- 证据入口：
  - `outputs/try/064_added_sota_added2_replay/seed_case_level_predictions.csv`
  - `outputs/try/064_added_sota_added2_replay/seed_summary_by_domain.csv`
  - `outputs/try/064_added_sota_added2_replay/stability_overview_by_domain.csv`
  - `outputs/try/064_added_sota_added2_replay/direct_2s_8s_summary_by_domain.csv`
  - `outputs/try/064_added_sota_added2_replay/comparison_overview.csv`
  - `outputs/try/064_added_sota_added2_replay/added2_case_comparison.csv`
  - `outputs/try/064_added_sota_added2_replay/model_assets.csv`
  - `outputs/try/064_added_sota_added2_replay/summary.md`

## 1. 目标

复核两个仓库里容易混在一起的 SOTA 口径是否能外推到 `added2`：

- `final SOTA`
  - `TinyTCN 2s+8s late fusion`
  - 直接复用 `063` 已经持久化的 full final deploy 输出
- `added 上界 SOTA`
  - `rpm_knn4 + TinyTCN all_channels midband(3.0-6.0Hz) @ w=0.3`
  - 复用 `042` 的 10 个 seed 与训练口径

## 2. 方法口径

### 2.1 `2s+8s`

- 不在 `064` 中重训；
- 复用 `063` 的 `case_level_predictions.csv`；
- 对照项包括：
  - `direct_tinytcn_2s_from_063`
  - `direct_tinytcn_8s_from_063`
  - `direct_tinytcn_2s_8s_fusion_from_063`

### 2.2 `midband`

- 训练池：`final` 带标签工况；
- eval：`added` 与 `added2`；
- 信号列：取 `final / added / added2` 的共同列；
- 输入：所有应变列带通到 `3.0-6.0Hz`，并拼接加速度列；
- 窗长：`5s`，即 `250` 点，步长 `125`；
- base seeds：
  - `42, 52, 62, 72, 82, 92, 102, 112, 122, 132`
- 对照项：
  - `rpm_knn4`
  - `tinytcn_all_channels_midband_3_0_6_0hz`
  - `fusion_rpm_knn4__tinytcn_all_channels_midband__w0.3`
  - `fusion_rpm_knn4__tinytcn_all_channels_midband__w0.5`
  - `fusion_rpm_knn4__tinytcn_all_channels_midband__w0.7`

## 3. 当前结果

### 3.1 [2026-04-08] added 上界 SOTA 在旧 `added` 上复现，但没有迁移到 `added2`

旧 `added` 复核：

- `fusion @ w=0.5`
  - `case_mae mean = 0.1592`
- `fusion @ w=0.3`
  - `case_mae mean = 0.1596`
- `rpm_knn4`
  - `case_mae mean = 0.2293`

`added2` 回放：

- `rpm_knn4`
  - `case_mae mean = 0.8131`
- `fusion @ w=0.3`
  - `case_mae mean = 1.0126`
  - `case_mae std = 0.0378`
- `fusion @ w=0.5`
  - `case_mae mean = 1.1456`
  - `case_mae std = 0.0630`
- `fusion @ w=0.7`
  - `case_mae mean = 1.2786`
  - `case_mae std = 0.0882`

这说明：

- `midband` 旧 added 收益不是脚本复现问题；
- `midband` 在 `added2` 上没有提供稳定正增益；
- 权重越大，`added2` 的整体误差越高。

### 3.2 [2026-04-08] `2s+8s` direct learned 仍明显弱于 rpm-first 路线

`added2` 对照：

- `direct_tinytcn_2s_from_063`
  - `case_mae = 2.1985`
- `direct_tinytcn_2s_8s_fusion_from_063`
  - `case_mae = 2.2897`
- `direct_tinytcn_8s_from_063`
  - `case_mae = 2.4157`

这说明：

- `2s+8s` 即使作为 final SOTA，也不是 `added2` 的默认外推候选；
- `added2` 上更合理的默认主干仍是 `rpm-first`。

### 3.3 [2026-04-08] added2 的 case 级结构解释了两个候选的互补与失败

重点 case：

- `工况25`
  - `direct 2s+8s abs_error = 0.5160`
  - `rpm_knn4 abs_error = 1.5985`
  - `midband @ w=0.3 abs_error = 2.0477`
- `工况26`
  - `direct 2s+8s abs_error = 0.2353`
  - `rpm_knn4 abs_error = 1.5949`
  - `midband @ w=0.3 abs_error = 1.6608`
- `工况28`
  - `rpm_knn4 abs_error = 0.1680`
  - `midband @ w=0.3 abs_error = 0.4863`
  - `direct 2s+8s abs_error = 3.0868`
- `工况29`
  - `rpm_knn4 abs_error = 0.2717`
  - `midband @ w=0.3 abs_error = 0.6116`
  - `direct 2s+8s abs_error = 3.2842`
- `工况30`
  - `rpm_knn4 abs_error = 0.1635`
  - `midband @ w=0.3 abs_error = 0.3105`
  - `direct 2s+8s abs_error = 5.0797`

这说明：

- `direct 2s+8s` 只在 `工况25-26` 有明显优势；
- `rpm_knn4` 在 `工况28-30` 明显更稳；
- `midband` 对 `工况27` 有轻微改善，但不够抵消 `工况25-26` 和 `工况28-30` 的拖累。

## 4. 模型资产

`064` 新增了 midband deploy checkpoint，入口为：

- `outputs/try/064_added_sota_added2_replay/model_assets.csv`
- `outputs/try/064_added_sota_added2_replay/models/checkpoints/final_deploy_midband_seed*_w5s.pt`
- `outputs/try/064_added_sota_added2_replay/models/checkpoints/final_deploy_midband_seed*_w5s_norm.npz`
- `outputs/try/064_added_sota_added2_replay/models/checkpoints/final_deploy_midband_seed*_w5s.json`

`2s+8s` 资产复用：

- `outputs/try/063_final_late_fusion_added2_replay/models/checkpoints/final_deploy_2s.pt`
- `outputs/try/063_final_late_fusion_added2_replay/models/checkpoints/final_deploy_8s.pt`

## 5. 当前判断

截至 `2026-04-08`，`added` 上界 SOTA `rpm_knn4 + TinyTCN all_channels midband(3.0-6.0Hz) @ w=0.3` 没有自然迁移到 `added2`；`added2` 整体上仍以 `rpm_knn4` 更稳。`direct 2s+8s` 对 `工况25-26` 有局部优势，但在 `工况28-30` 上失效明显，因此不能作为 `added2` 默认主线。
