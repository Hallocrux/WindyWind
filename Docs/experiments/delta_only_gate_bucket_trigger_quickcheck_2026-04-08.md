# delta-only gate / bucket / trigger quickcheck（2026-04-08）

- 状态：historical
- 首次确认：2026-04-08
- 最近复核：2026-04-08
- 数据范围：
  - `final` 代表性 holdout：`工况1 / 3 / 17 / 18`
  - `added`：`工况21-24`
- 代码口径：
  - `src/try/059_delta_only_gate_bucket_trigger_quickcheck/`
- 证据入口：
  - `outputs/try/059_delta_only_gate_bucket_trigger_quickcheck/summary.md`
  - `outputs/try/059_delta_only_gate_bucket_trigger_quickcheck/summary_by_domain.csv`
  - `outputs/try/059_delta_only_gate_bucket_trigger_quickcheck/case_level_predictions.csv`
  - `outputs/try/059_delta_only_gate_bucket_trigger_quickcheck/gate_feature_table.csv`

## 1. 目标

- 延续 `058` 的方向：
  - `embedding_concat` 继续作为默认检索空间；
  - correction head 继续收缩到 `delta-only`；
- 在这个更保守的 correction 上，再验证三类路由：
  - `soft gate`
  - `bucket`
  - `trigger`
- 关注点是：
  - added 侧是否还能保留局部收益；
  - final 侧能否把 `058 delta-only` 的额外伤害压回去。

## 2. 实验设置

- base：
  - `rpm_knn4`
- enhanced candidate：
  - `rpm_knn4 + delta_only_prototype_ridge @ w=0.5`
- 路由输入特征：
  - prototype 特征：
    - `pred_gap / abs_pred_gap`
    - `top1_embed_distance`
    - `topk_embed_mean_distance`
    - `topk_embed_std_distance`
    - `delta_l2 / delta_l1_mean / delta_signed_mean / delta_max_abs`
  - 机制特征：
    - `strain_low_over_mid`
    - `strain_mid_ratio_median`
    - `strain_rms_median`
    - `acc_energy_median`
    - `acc_peak_freq_median`
    - `strain_acc_rms_ratio`
    - `missing_ratio_in_common_cols`
    - `edge_removed_ratio`
- 训练方式：
  - outer holdout 继续沿用 `052/056/058` 的 `8` 个代表性工况；
  - 对每个 outer fold 的训练池，再做 inner-LOO，构造 gate 训练标签。

## 3. 核心结果

### 3.1 汇总

- `rpm_knn4`
  - `final_focus case_mae = 1.0595`
  - `added_focus case_mae = 0.2293`
  - `focus_all case_mae = 0.6444`
- `058 | rpm_knn4 + delta_only_prototype_ridge @ w=0.5`
  - `final_focus case_mae = 1.1828`
  - `added_focus case_mae = 0.2612`
  - `focus_all case_mae = 0.7220`
- `059 | delta_only_trigger_rule_cv`
  - `final_focus case_mae = 1.0595`
  - `added_focus case_mae = 0.1932`
  - `focus_all case_mae = 0.6264`
- `059 | delta_only_two_stage_hgb_t0.65`
  - `added_focus case_mae = 0.1926`
  - `final_focus case_mae = 1.2024`
- `052 | rpm_knn4 + embedding_residual_concat_2s_8s @ w=0.5`
  - `final_focus case_mae = 1.0159`
  - `added_focus case_mae = 0.1852`
  - `focus_all case_mae = 0.6006`

### 3.2 每工况信号

- `trigger_rule_cv`
  - 对 `工况1 / 17 / 18 / 21 / 22 / 23` 退回 `rpm_knn4`
  - 只保留了 `工况24` 的修正收益
  - 没能稳定保留 `工况3` 的负向 correction 收益
- `soft_gate_hgb`
  - 能部分保留 `工况3 / 24` 的收益
  - 但仍会在 `工况1 / 17 / 18 / 22` 上重新引入额外伤害
- `bucket / two-stage`
  - added 侧会比 base 更好
  - 但 final 侧回退不够保守，重新吃回 `058 delta-only` 的伤害

## 4. 解释

- `delta-only` 的主要价值在于：
  - 把 `056` 的 full head 自由度收住；
  - 让 correction 可以进入“只在少数样本触发”的阶段。
- 这一轮更支持：
  - `delta-only` 应该被当成“候选修正”，不是默认总是启用的增强分支；
  - 在小样本条件下，`trigger` 比连续 `soft gate` 更稳；
  - `bucket / two-stage` 虽然能保留部分 added 改善，但 final 侧仍不够安全。

## 5. 当前可保留结论

- `embedding_concat` 继续适合作为默认检索空间；
- correction head 继续应以 `delta-only` 为主；
- 如果下一步目标是“保住 added 的局部收敛，同时把 final 额外伤害再压低”，优先顺序更适合是：
  - `delta-only + conservative trigger`
  - 再考虑更细的 bucket
  - 不宜直接回到连续 residual gate 或更自由的 head。

## 6. 当前限制

- 该结果只覆盖代表性 `8` 工况 quickcheck；
- `trigger_rule_cv` 还没有稳定找回 `工况3` 这类“应当启用负向 correction”的 final hard case；
- 因此它更适合作为下一步默认研究方向，而不是直接升级成统一主线。
