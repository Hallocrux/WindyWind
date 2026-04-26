# added 异质处理后的分支路线图（2026-04-07）

- 状态：`current`
- 首次确认：`2026-04-07`
- 最近复核：`2026-04-07`
- 数据范围：
  - `data/added/` 的 `工况21-24`
  - 对照域：`data/final/` 的带标签工况
- 证据入口：
  - `PROJECT.md`
  - `Docs/experiments/added_validation_label_check_2026-04-07.md`
  - `Docs/experiments/added_domain_diagnosis_2026-04-07.md`
  - `Docs/experiments/case22_label_and_modality_check_2026-04-07.md`
  - `Docs/experiments/strain_shift_mitigation_check_2026-04-07.md`
  - `Docs/experiments/strain_transfer_band_scan_2026-04-07.md`
  - `Docs/experiments/midband_strain_weight_scan_2026-04-07.md`
  - `Docs/experiments/rpm_learned_midband_multiseed_stability_check_2026-04-07.md`
  - `Docs/experiments/pred_rpm_deployability_check_2026-04-07.md`
  - `Docs/experiments/fft_rpm_algorithm_search_2026-04-07.md`
  - `Docs/experiments/fft_rpm_to_wind_replay_2026-04-07.md`
  - `Docs/experiments/fft_midband_fusion_replay_2026-04-07.md`
  - `Docs/route_gate_strategy_2026-04-07.md`

## 1. 这条线为什么会突然分叉

`2026-04-07` 的 `034-036` 连续验证之后，`added` 方向不再是一个单一问题，而是被拆成了三层：

1. `added` 是否只是标签问题；
2. `added` 的失配到底由哪类模态驱动；
3. 如果真实 `rpm` 不能部署，最终该保留哪条可部署路线。

从这一刻开始，后续 try 的本质不再是“继续调一个模型”，而是分别在回答不同子问题，所以看起来像很多优化分支同时长出来。

## 2. 分支演化主线

| 阶段 | 对应 try | 当时回答的问题 | 当前沉淀下来的结论 |
| --- | --- | --- | --- |
| A. 先确认是不是训练池或标签锅 | `034-036` | `added` 高估是不是因为 clean/full pool 选错、通道被裁坏、或标签本身错了 | 不是训练池污染主导，也不是共有通道收缩；`added` 更像额外分布，`工况22` 是单独异常机制点 |
| B. 再定位是哪类信号坏掉 | `037-038` | 是加速度坏，还是应变坏；应变还能不能救 | 主问题在应变侧，尤其是低频；原始 `all_channels` 和 `strain_only` 都不该再当默认路线 |
| C. 把“坏应变”缩成“有用应变” | `039-040` | 应变里还有没有可迁移的频带 | 有，且已收敛到 `3.0-6.0Hz`；它不是全频应变的替代，而是一个可保留的 learned 中频支线 |
| D. 把解析支线和 learned 支线重新拼起来 | `041-042` | `rpm_knn4` 和 learned midband 是替代还是互补 | 是互补；当前 `true_rpm` 上界参考里更稳的是 `rpm_knn4 + TinyTCN all_channels midband @ w≈0.3` |
| E. 把“上界”改写成“可部署” | `043 / 043_1 / 043_2 / 043_3` | 没有真实 `rpm` 时还能不能保住 added 增益 | `pred_rpm` 支线在 added 上崩坏，不能进默认矩阵；FFT 解析 RPM 可以替代 `true_rpm` 进入 added deployable 融合线 |
| F. 统一主线还是双轨 | `044+` | `final + added` 能否共用一条默认主线 | 截至 `2026-04-07` 仍未收敛；更像要做 `route / gate`，而不是强行统一 |

## 3. 当前看这些分支，应该怎么归类

### 3.1 已经基本淘汰或降级的分支

- `added` 直接拿来做标签审计：
  - `034` 已说明，在外部失配没解释前，`added 21-24` 不能直接充当稳定审标签集。
- “难工况污染训练池”解释：
  - `035` 已说明，把 `工况1 / 3 / 17 / 18` 加回训练池只会让 `added` 更糟。
- 原始 `all_channels` 默认外推路线：
  - `037-038` 已说明，added 高估主驱动是应变侧，原始全频应变不能直接回到默认主线。
- `pred_rpm -> wind` 作为 added 默认解析支线：
  - `043` 已说明，它在 `final` 域内还能工作，但在 `added` 域明显塌到高转速区，不应进入默认候选矩阵。
- “先做 final/added 二分类”：
  - `route_gate_strategy` 已把问题重写成“该走哪条路线”，不推荐先把目录来源当监督标签。

### 3.2 当前仍值得保留的主分支

- 解析基线分支：
  - 上界参考是 `true_rpm -> rpm_knn4`；
  - added 可部署替代已切到 `fft_peak_1x_whole -> rpm_knn4`。
- learned 中频支线：
  - 当前保留形式是 `TinyTCN all_channels + strain bandpass 3.0-6.0Hz`；
  - 它的角色是“增强分支”，不是全局默认主干。
- 解析 + learned 融合分支：
  - 上界参考：`rpm_knn4 + learned midband @ w≈0.3`；
  - deployable added 候选：`fft_peak_1x_whole -> rpm_knn4 + learned midband @ w≈0.3`。

### 3.3 还没收口、但值得继续的分支

- `044 final LOCO FFT + learned midband`：
  - 截至 `2026-04-07` 只落了前 `3` 个 seed，中间结果更像“added 增益未自动迁移到 final”。
- `route / gate quickcheck`：
  - 如果 `final` 和 `added` 继续分化，下一步重点就不是继续扩模型，而是学会“什么时候开增强分支”。
- FFT `142 rpm` 吸附消歧：
  - `043_1` 已经暴露出这是解析支线在旧域里的主要失败模式。
- `工况22` 标签证据工件补齐：
  - `037` 已说明它不是明显假标签，但仓库里缺少截图工件，证据链还不完整。

## 4. 现在回头看，真正的决策树其实很短

### 4.1 如果目标只是“解释 added 为什么坏了”

结论已经收口为：

- 不是训练池污染；
- 不是通道口径变化；
- 主因是应变侧低频域偏移；
- `工况22` 是最强异常点。

### 4.2 如果目标只是“在 added 上做当前最稳的可部署方案”

当前默认参考应写成：

- `fft_peak_1x_whole -> rpm_knn4 + TinyTCN all_channels midband(3.0-6.0Hz) @ learned_weight≈0.3`

### 4.3 如果目标是“final 保稳，added 也想吃到增强收益”

当前不应直接问：

- “要不要把 added 那套直接升级成全局默认主线”

而应改问：

- “什么时候走 base route，什么时候走 enhanced route”

也就是：

- 先补齐 `044/045` 的双域复核；
- 如果双域仍冲突，就进入 `route / gate`。

## 5. 推荐把脑图压缩成这三个层级

以后再看这条线，可以只记下面三层，不必把 `034-043_3` 当成十几个并列分支。

### 5.1 第一层：诊断层

- `034-036`
- 任务：证明这是域偏移，不是简单标签错或训练池错

### 5.2 第二层：修复层

- `037-040`
- 任务：把“坏应变”缩成“可迁移的 `3.0-6.0Hz` 中频应变”

### 5.3 第三层：部署层

- `041-043_3`
- 任务：把“解析支线 + learned 中频支线”做成 added 可部署候选，并淘汰 `pred_rpm`

## 6. 一句话版结论

截至 `2026-04-07`，`added` 异质之后长出来的很多分支，其实已经可以压缩成一句话：

不是“每条路都还活着”，而是已经收敛成了“放弃原始全频应变与 `pred_rpm` added 部署线，保留 `FFT 解析 RPM + 3.0-6.0Hz learned 中频应变`，然后再决定是否需要 `route / gate` 来兼容 `final + added` 双域”。
