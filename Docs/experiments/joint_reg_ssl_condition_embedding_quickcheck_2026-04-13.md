# joint reg ssl condition embedding quickcheck（2026-04-13）

- 状态：`current`
- 首次确认：`2026-04-13`
- 最近复核：`2026-04-13`
- 数据范围：
  - encoder 联合训练：`final` 带标签工况 `1, 3-20` + `added` 带标签工况 `21-24`
  - residual 训练：`data/added/` 的带标签工况 `21-24`
  - 测试：`data/added2/` 的带标签工况 `25-30`
- 代码口径：
  - `src/try/087_joint_reg_ssl_condition_embedding/`
- 证据入口：
  - `outputs/try/087_joint_reg_ssl_condition_embedding/summary_by_protocol.csv`
  - `outputs/try/087_joint_reg_ssl_condition_embedding/case_level_predictions.csv`
  - `outputs/try/087_joint_reg_ssl_condition_embedding/embedding_case_table.csv`
  - `outputs/try/087_joint_reg_ssl_condition_embedding/knn_neighbors.csv`
  - `outputs/try/087_joint_reg_ssl_condition_embedding/summary.md`
- 模型资产：
  - `outputs/try/087_joint_reg_ssl_condition_embedding/models/checkpoints/joint_2s_lambda0p1.pt`
  - `outputs/try/087_joint_reg_ssl_condition_embedding/models/checkpoints/joint_2s_lambda0p1_norm.npz`
  - `outputs/try/087_joint_reg_ssl_condition_embedding/models/checkpoints/joint_2s_lambda0p1.json`
  - `outputs/try/087_joint_reg_ssl_condition_embedding/models/checkpoints/joint_8s_lambda0p1.pt`
  - `outputs/try/087_joint_reg_ssl_condition_embedding/models/checkpoints/joint_8s_lambda0p1_norm.npz`
  - `outputs/try/087_joint_reg_ssl_condition_embedding/models/checkpoints/joint_8s_lambda0p1.json`

## 1. 目标

验证联合训练口径：

```text
L = L_reg(wind_speed) + lambda_ssl * L_ssl(case structure)
```

本轮使用：

- `lambda_ssl = 0.1`
- `L_reg`：窗口级风速 MSE；
- `L_ssl`：按 `case_id` 组织的 supervised contrastive loss；
- embedding 下游评估继续沿用 `added -> added2` 的 residual ridge 口径。

## 2. 当前结果

### 2.1 [2026-04-13] 联合训练 V1 优于 contrastive-only V1，但仍弱于 `071`

`added_to_added2` 结果：

- `rpm_knn4 + joint_embedding_residual_ridge`
  - `case_mae = 0.8935`
  - `case_rmse = 1.0235`
  - `max_abs_error = 1.7023`
- `rpm_knn4`
  - `case_mae = 1.2903`
  - `case_rmse = 1.5511`
  - `max_abs_error = 2.9911`
- `joint_embedding_ridge`
  - `case_mae = 1.4391`
  - `case_rmse = 1.5793`
  - `max_abs_error = 2.4825`

对照：

- `086 | rpm_knn4 + contrastive_embedding_residual_ridge`
  - `case_mae = 0.9264`
- `071 | rpm_knn4 + embedding residual ridge`
  - `case_mae = 0.6161`

这说明：

- 加入 `L_reg` 后，joint embedding 比 contrastive-only embedding 更适合当前 residual 任务；
- 但当前 `lambda_ssl = 0.1` 的 V1 仍未超过 `071`；
- 因此 `087` 当前不能替代 `071` 成为 added-first 默认最佳模型。

### 2.2 [2026-04-13] 当前主要收益仍集中在高风速端，低风速端仍有过冲

逐工况对照显示：

- `工况25`
  - `rpm_knn4 abs_error = 2.9911`
  - `rpm_knn4 + joint residual abs_error = 1.7023`
- `工况26`
  - `rpm_knn4 abs_error = 1.7651`
  - `rpm_knn4 + joint residual abs_error = 0.3389`
- `工况27`
  - `rpm_knn4 abs_error = 1.0240`
  - `rpm_knn4 + joint residual abs_error = 0.3907`
- `工况28`
  - `rpm_knn4 abs_error = 0.4901`
  - `rpm_knn4 + joint residual abs_error = 0.8775`
- `工况29`
  - `rpm_knn4 abs_error = 0.7685`
  - `rpm_knn4 + joint residual abs_error = 1.3766`
- `工况30`
  - `rpm_knn4 abs_error = 0.7032`
  - `rpm_knn4 + joint residual abs_error = 0.6749`

这说明：

- joint residual 对 `工况25-27` 的高风速端修正强于 `086`；
- 但对 `工况28-29` 的低风速端过冲仍未解决；
- 后续如果继续做联合训练，更值得优先做 `lambda_ssl` 扫描与 residual gate，而不是直接把当前 V1 升级为默认线。

## 3. 当前判断

截至 `2026-04-13`，这轮 quickcheck 支持下面的表达：

- `L_reg + lambda_ssl * L_ssl` 是可运行路线；
- 在同一 `added -> added2` 口径下，`087` 优于 `086`；
- 但 `087` 仍明显弱于 `071`；
- 当前默认 added-first 最佳模型仍应保持 `071 | rpm_knn4 + embedding residual ridge`。
