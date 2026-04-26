# prototype alignment SOTA recheck（2026-04-08）

- 状态：historical
- 首次确认：2026-04-08
- 最近复核：2026-04-08
- 数据范围：
  - `final`：带标签工况全量 `LOCO`
  - `added`：`工况21-24`
- 代码口径：
  - `src/try/061_prototype_alignment_sota_recheck/`
- 证据入口：
  - `outputs/try/061_prototype_alignment_sota_recheck/final_summary.csv`
  - `outputs/try/061_prototype_alignment_sota_recheck/added_summary.csv`
  - `outputs/try/061_prototype_alignment_sota_recheck/comparison_to_sota.csv`
  - `outputs/try/061_prototype_alignment_sota_recheck/summary.md`

## 1. 目标

- 不再看代表性 holdout quickcheck；
- 直接把 `060 prototype alignment` 当前版放到两个目标口径上对照：
  - `final SOTA`
  - `added 上界 SOTA`

## 2. 对照目标

### 2.1 final SOTA

- 参考变体：
  - `TinyTCN_multiscale_late_fusion_2s_8s`
- 参考口径：
  - `full final LOCO`
- 参考指标：
  - `case_mae = 0.1685`

### 2.2 added 上界 SOTA

- 参考变体：
  - `rpm_knn4 + TinyTCN all_channels midband @ w=0.3`
- 参考口径：
  - `full_final_pool -> added 21-24`
- 参考指标：
  - `case_mae mean = 0.1627`

## 3. 当前版结果

### 3.1 final 全量 LOCO

- `rpm_knn4`
  - `case_mae = 0.4256`
- `prototype_alignment_ridge @ w=0.5`
  - `case_mae = 0.4904`
  - 相对 final SOTA：
    - `gap = +0.3219`
- `prototype_alignment_ridge`
  - `case_mae = 0.6634`
  - 相对 final SOTA：
    - `gap = +0.4949`

### 3.2 added 外部评估

- `rpm_knn4`
  - `case_mae = 0.2293`
- `prototype_alignment_ridge @ w=0.5`
  - `case_mae = 0.4410`
  - 相对 added 上界：
    - `gap = +0.2783`
- `prototype_alignment_ridge`
  - `case_mae = 0.9213`
  - 相对 added 上界：
    - `gap = +0.7586`

## 4. case 级观察

### 4.1 final

- `w=0.5` 的主要拖后腿工况：
  - `工况1`
    - `abs_error = 2.1939`
  - `工况17`
    - `abs_error = 1.9567`
  - `工况3`
    - `abs_error = 1.0690`
  - `工况18`
    - `abs_error = 0.8303`

### 4.2 added

- `w=0.5` 的 case 级误差：
  - `工况21`
    - `abs_error = 0.6689`
  - `工况22`
    - `abs_error = 0.3438`
  - `工况23`
    - `abs_error = 0.3813`
  - `工况24`
    - `abs_error = 0.3700`

## 5. 当前判断

- `060` 在代表性 hard-case quickcheck 中出现的正信号，没有迁移成同口径 SOTA 结果；
- 放到 `full final LOCO` 后，当前版甚至没有超过 `rpm_knn4`；
- 放到 `added 21-24` 后，当前版与 added 上界差距更明显；
- 因此到 `2026-04-08` 这一步，更合理的结论是：
  - `prototype alignment` 仍是一个值得分析的表征层方向；
  - 但“当前版实现”还没有到达 `final SOTA`，也没有到达 `added 上界 SOTA`；
  - 在没有进一步约束前，它还不适合继续作为统一主候选推进。
