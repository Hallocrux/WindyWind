# embedding top-k prototype alignment quickcheck（2026-04-08）

- 状态：historical
- 首次确认：2026-04-08
- 最近复核：2026-04-08
- 数据范围：
  - `final` 代表性 holdout：`工况1 / 3 / 17 / 18`
  - `added`：`工况21-24`
- 代码口径：
  - `src/try/060_embedding_topk_prototype_alignment_quickcheck/`
- 证据入口：
  - `outputs/try/060_embedding_topk_prototype_alignment_quickcheck/summary.md`
  - `outputs/try/060_embedding_topk_prototype_alignment_quickcheck/summary_by_domain.csv`
  - `outputs/try/060_embedding_topk_prototype_alignment_quickcheck/case_level_predictions.csv`
  - `outputs/try/060_embedding_topk_prototype_alignment_quickcheck/alignment_feature_table.csv`

## 1. 目标

- 明确按“新的融合”定义推进：
  - `embedding top-k` 只负责检索局部参考域；
  - 先把 top-k 邻居融合成 `local prototype`；
  - 再在表征层比较 / 对齐 `target` 与 `prototype`；
  - 最后只做 bounded、保守的修正。
- 本实验显式避免：
  - 邻居风速平均；
  - 邻居 residual 平均。

## 2. 方法摘要

- `retrieve`
  - 使用 `TinyTCN 2s+8s case embedding concat`
  - 在训练池里做 `top-k`
- `prototype`
  - 距离加权均值构造 `h_ref`
  - 同时统计 support 的逐维离散度
- `alignment`
  - `delta = h - h_ref`
  - 用局部离散度把 `delta` 归一化成 `delta_z`
  - 再对 `delta_z` 做 `tanh` 收缩，得到 `aligned target`
- `prediction`
  - 只使用：
    - `prototype` 的压缩坐标
    - `aligned target` 相对 `prototype` 的压缩位移
    - prototype 可信度 / 离散度统计
  - 输出 bounded correction

## 3. 核心结果

### 3.1 汇总

- `rpm_knn4`
  - `final_focus case_mae = 1.0595`
  - `added_focus case_mae = 0.2293`
  - `focus_all case_mae = 0.6444`
- `052 | rpm_knn4 + embedding_residual_concat_2s_8s @ w=0.5`
  - `final_focus case_mae = 1.0159`
  - `added_focus case_mae = 0.1852`
  - `focus_all case_mae = 0.6006`
- `058 | delta_only_prototype_ridge @ w=0.5`
  - `final_focus case_mae = 1.1828`
  - `added_focus case_mae = 0.2612`
  - `focus_all case_mae = 0.7220`
- `060 | embedding_prototype_alignment_ridge`
  - `final_focus case_mae = 0.7951`
  - `added_focus case_mae = 0.8009`
  - `focus_all case_mae = 0.7980`
- `060 | embedding_prototype_alignment_ridge @ w=0.5`
  - `final_focus case_mae = 0.9262`
  - `added_focus case_mae = 0.3537`
  - `focus_all case_mae = 0.6399`

### 3.2 每工况信号

- `final_focus`
  - `工况1`
    - `rpm_knn4`: `1.5026`
    - `alignment_ridge`: `1.0238`
  - `工况3`
    - `rpm_knn4`: `1.1322`
    - `alignment_ridge`: `0.6853`
  - `工况17`
    - `rpm_knn4`: `1.5987`
    - `alignment_ridge`: `1.4018`
  - `工况18`
    - `rpm_knn4`: `0.0046`
    - `alignment_ridge @ w=0.5`: `0.0325`
- `added_focus`
  - `工况21`
    - `alignment_ridge @ w=0.5`: `0.1645`
  - `工况22`
    - `alignment_ridge @ w=0.5`: `0.2623`
  - `工况23`
    - `alignment_ridge @ w=0.5`: `0.2628`
  - `工况24`
    - `alignment_ridge @ w=0.5`: `0.7251`

## 4. 解释

- 这说明“先构造 prototype，再在表征层做 alignment”这条线不是无效的：
  - 对 `final` hard case，prototype-relative 高维信号明显比简单 `rpm_knn4` 更有解释力；
  - 尤其 `工况1 / 3 / 17` 出现了稳定改善。
- 但这条线当前仍不够适合作为 unified 默认候选：
  - added 域会被 prototype alignment 过度牵引；
  - `工况24` 的局部参考域里混入了 `final` 邻居后，alignment head 出现了明显误拉。

## 5. 当前可保留结论

- “新的融合”定义本身是成立的：
  - `top-k -> prototype -> alignment -> bounded prediction`
  - 这条线和“邻居 residual 平均”不是一回事；
  - 它已经在 `final` hard case 上表现出独立正信号。
- 当前主要问题不是 prototype 本身无效，而是：
  - added 外部域上的 prototype 可信度还不够；
  - 缺少进一步的 domain gate / trust gate。

## 6. 当前限制

- 该结果只覆盖 `8` 工况 quickcheck；
- 当前 alignment head 还没有叠加专门的 added 保守触发；
- 因此这条线当前更适合作为：
  - `final` hard case 的强候选方向；
  - 以及后续“prototype alignment + trust gate”组合路线的前置证据。
