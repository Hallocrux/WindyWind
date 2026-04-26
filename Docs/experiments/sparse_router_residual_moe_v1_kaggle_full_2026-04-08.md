# sparse router residual MoE V1 Kaggle full run（2026-04-08）

- 状态：`current`
- 首次确认：`2026-04-08`
- 最近复核：`2026-04-08`
- 数据范围：
  - `final`：带标签工况全量 `LOCO`
  - `added`：`工况21-24`
- 代码口径：
  - `src/try/065_sparse_router_residual_moe_v1/`
- 证据入口：
  - `outputs/kaggle_publish/065_sparse_router_residual_moe_v1/kernel_output_v3_retry/windywind/outputs/try/065_sparse_router_residual_moe_v1/summary_by_domain.csv`
  - `outputs/kaggle_publish/065_sparse_router_residual_moe_v1/kernel_output_v3_retry/windywind/outputs/try/065_sparse_router_residual_moe_v1/summary.md`
  - `outputs/kaggle_publish/065_sparse_router_residual_moe_v1/kernel_output_v3_retry/windywind/outputs/try/065_sparse_router_residual_moe_v1/router_activation_table.csv`
  - `outputs/kaggle_publish/065_sparse_router_residual_moe_v1/kernel_output_v3_retry/windywind/outputs/try/065_sparse_router_residual_moe_v1/prototype_retrieval_stats.csv`
  - `outputs/kaggle_publish/065_sparse_router_residual_moe_v1/kernel_output_v3_retry/windywind/outputs/try/065_sparse_router_residual_moe_v1/static_checks.csv`

## 1. 目标

在 `2026-04-08` 已有 `gate / embedding residual / prototype` 证据基础上，做一次真正的统一训练版 `MoE V1` 全量复核，回答：

- `rpm-first + shared encoder + sparse router + bounded residual experts` 这条 unified 路线是否值得继续；
- 该结构在 `final` 域内能否给出独立正信号；
- `L_noharm` 是否真的在保护 `final`；
- 该结构在 `added external` 上是否已经具备默认增强价值。

## 2. 方法口径

### 2.1 结构

- `base = rpm_knn4`
- shared encoder：
  - `TinyTCN 2s+8s`
- experts：
  - `Expert0 = no-op`
  - `Expert1 = global residual expert`
  - `Expert2 = prototype delta-only residual expert`
- router：
  - 一次性 sparse `top-k`
- 所有 expert：
  - 只输出 `bounded residual`

### 2.2 损失

- `L_main = Huber(pred, y)`
- `L_noharm = max(0, |pred-y| - |base_pred-y| - margin)`
- `L_delta = mean(sum_e alpha_e * |delta_e|)`

### 2.3 对照组

- `A0 = rpm_knn4`
- `A1 = rpm_knn4 + global residual only`
- `A2 = rpm_knn4 + prototype delta-only residual only`
- `A3 = sparse router + 3 experts`
- `A4 = A3 without L_noharm`

### 2.4 初始化

- `final LOCO`：
  - 对 `工况1 / 3 / 17 / 18` 复用 `053` 已落盘 fold checkpoint
  - 其余 `final` 工况回退到 random init
- `added external`：
  - 复用 `063` 的 `final_deploy_2s / 8s` checkpoint 做部分初始化

## 3. 当前结果

### 3.1 [2026-04-08] `A3_sparse_router_moe` 在 `final LOCO` 上给出了弱正信号

`final` 域对照：

- `A3_sparse_router_moe`
  - `case_mae = 0.4056`
  - `case_rmse = 0.6173`
- `A0_rpm_knn4`
  - `case_mae = 0.4256`
  - `case_rmse = 0.6696`
- `A2_prototype_delta_only`
  - `case_mae = 0.4243`
- `A1_global_residual_only`
  - `case_mae = 0.4297`

这说明：

- unified sparse residual-MoE 在 `final` 域内不是完全无效；
- `router + residual experts` 的联合写法已经能略微超过 `rpm_knn4`；
- `A3` 也优于单独 `A1 / A2`，说明 unified routing 本身有额外价值。

### 3.2 [2026-04-08] 当前版 `MoE V1` 在 `added external` 上明显不如 `rpm_knn4`

`added` 域对照：

- `A0_rpm_knn4`
  - `case_mae = 0.2293`
- `A4_sparse_router_moe_without_noharm`
  - `case_mae = 0.2293`
- `A2_prototype_delta_only`
  - `case_mae = 0.4895`
- `A3_sparse_router_moe`
  - `case_mae = 0.4955`
- `A1_global_residual_only`
  - `case_mae = 0.5453`

这说明：

- 当前 unified sparse residual-MoE 还不能作为 `added` 默认增强器；
- `added` 上的主要问题不是“没有修正能力”，而是“修正过强且不够可信”；
- 就当前版而言，`added` 上更稳的默认选择仍是 `rpm_knn4`。

### 3.3 [2026-04-08] `L_noharm` 在 `final` 上确实有效

`final` 域对照：

- `A3_sparse_router_moe`
  - `worse_than_base_rate = 0.3158`
  - `mean_excess_harm = 0.0165`
- `A4_sparse_router_moe_without_noharm`
  - `worse_than_base_rate = 0.6316`
  - `mean_excess_harm = 0.0732`

这说明：

- `L_noharm` 不是形式化装饰，而是在当前训练里确实产生了保护作用；
- 去掉它之后，`final` 伤害显著放大；
- 若后续继续做 unified MoE，这个保护项应继续保留，而不是移除。

### 3.4 [2026-04-08] 当前 router 行为在 `final` 上已有部分保守性，但在 `added` 上仍过于激进

`A3_sparse_router_moe` 的平均路由行为：

- `final`
  - `no-op = 0.3158`
  - `global = 0.0000`
  - `prototype = 0.6842`
- `added`
  - `no-op = 0.0000`
  - `global = 0.0000`
  - `prototype = 1.0000`

这说明：

- 在 `final` 上，router 已经学会让一部分样本退回 `no-op`；
- 但在 `added` 上，它几乎把全部样本都推给了 `prototype expert`；
- 当前 added 退化的主因更接近“prototype expert 外部域 trust 不足”，而不是“router 没学到任何结构”。

### 3.5 [2026-04-08] 当前静态检查全部通过，失败来自模型行为，不是实现错误

`static_checks.csv` 显示：

- `router_topk_respected = 1`
- `expert0_is_zero_output = 1`
- `expert2_delta_only_path_present = 1`

这说明：

- 这次结果的偏差主要是模型行为问题；
- 不是因为 `top-k` 没生效；
- 也不是因为 `Expert0` 或 `Expert2` 实现偏离了 `V1` 设计稿。

## 4. 当前判断

截至 `2026-04-08`，这次 Kaggle GPU full run 更支持下面的表达：

- `MoE V1` 已经证明 unified sparse residual-MoE 不是死路；
- 它在 `final` 域内已经出现独立弱正信号；
- `L_noharm` 的保护作用已经被真正复核到；
- 但当前版在 `added external` 上仍明显不如 `rpm_knn4`；
- 因此这次运行不能被解读为：
  - `MoE V1` 已经成为统一默认主线

更合理的结论是：

- 当前版 `MoE V1` 更像“主域内有潜力、外部域仍缺 trust / abstain 机制”的研究型原型；
- 下一步若继续沿 unified MoE 迭代，优先级应放在：
  - 更强的 external abstain / trust gate
  - 而不是继续扩 expert 数量。

## 5. 一句话版结论

截至 `2026-04-08`，`065` 的 Kaggle GPU full run 已经证明 unified sparse residual-MoE 在 `final LOCO` 上存在可保留的正信号，且 `L_noharm` 确实有效；但当前版在 `added external` 上仍明显劣于 `rpm_knn4`，因此它还不能升级为统一默认主线，只能保留为下一步继续加 trust / abstain 约束的研究型候选。
