# FFT RPM 与 learned midband 融合回放（2026-04-07）

- 状态：`current`
- 首次确认：`2026-04-07`
- 最近复核：`2026-04-07`
- 数据范围：
  - `data/added/` 的 `工况21-24`
- 代码口径：
  - `src/try/043_3_fft_midband_fusion_replay/`
- 证据入口：
  - `outputs/try/043_3_fft_midband_fusion_replay/case_level_predictions.csv`
  - `outputs/try/043_3_fft_midband_fusion_replay/seed_summary.csv`
  - `outputs/try/043_3_fft_midband_fusion_replay/stability_overview.csv`
  - `outputs/try/043_3_fft_midband_fusion_replay/pairwise_comparison.csv`
  - `outputs/try/043_3_fft_midband_fusion_replay/best_variant_by_seed.csv`
  - `outputs/try/043_3_fft_midband_fusion_replay/summary.md`

## 1. 目标

在 `041/042` 已经证明 `true_rpm + learned midband` 稳定成立、`043_2` 已经证明 FFT 支线可以形成更强的可部署 `rpm -> wind` 链之后，进一步回答：

- FFT 求 RPM 能否替代 `true_rpm`，继续进入最终融合算法；
- 这种替代如果成立，与 `true_rpm` 上界相比还差多少；
- 替代后更稳的默认权重应落在什么位置。

## 2. 方法口径

- 不重新训练任何 `TinyTCN`；
- 不重新跑任何 FFT RPM 识别；
- 只复用两类已有落盘结果：
  - `042` 的 `TinyTCN all_channels midband` 多随机种子 case 级预测；
  - `043_2` 的 FFT deployable `rpm -> wind` case 级预测；
- 在本 try 内仅做：
  - case 级对齐；
  - 线性晚融合；
  - 多 seed 汇总比较。

本轮复用的解析分支候选为：

- `fft_fft_peak_1x_whole__to__rpm_knn4`
- `fft_hybrid_peak_1x_whole_window8_gate150__to__rpm_knn4`
- `fft_window_peak_1x_conf_8s__to__rpm_knn4`

learned 分支固定为：

- `tinytcn_all_channels_midband_3_0_6_0hz`

融合权重复用此前口径：

- `w = 0.3 / 0.5 / 0.7`

预测形式为：

- `pred = (1 - w) * pred_fft_branch + w * pred_learned_branch`

## 3. 当前结果

### 3.1 [2026-04-07] 当前最优 deployable 融合候选为 `fft_peak_1x_whole -> rpm_knn4 + learned midband @ w=0.3`

- 状态：`current`
- 首次确认：`2026-04-07`
- 最近复核：`2026-04-07`

多 seed 汇总结果：

- `fusion_fft_fft_peak_1x_whole__rpm_knn4__tinytcn_all_channels_midband__w0.3`
  - `case_mae mean = 0.1675`
  - `case_mae std = 0.0202`
  - `case22_abs_error mean = 0.0899`
- `fusion_fft_window_peak_1x_conf_8s__rpm_knn4__tinytcn_all_channels_midband__w0.5`
  - `case_mae mean = 0.1842`
- `fusion_fft_hybrid_peak_1x_whole_window8_gate150__rpm_knn4__tinytcn_all_channels_midband__w0.3`
  - `case_mae mean = 0.1854`

这说明：

- 在当前 added 可部署融合链里，最稳的解析输入不再是 `true_rpm`，而是 `fft_peak_1x_whole -> rpm_knn4`；
- learned 权重依然不宜过重，当前仍更支持保守的 `0.3` 邻域；
- `window_peak_1x_conf_8s` 虽然不是全局均值最优，但在部分 seed 上仍能成为局部最优候选。

### 3.2 [2026-04-07] FFT 替代 `true_rpm` 后，added 融合效果已经非常接近原上界

- 状态：`current`
- 首次确认：`2026-04-07`
- 最近复核：`2026-04-07`

关键对照：

- `true_rpm` 上界参考
  - `fusion_rpm_knn4__tinytcn_all_channels_midband__w0.3`
  - `case_mae mean = 0.1627`
  - `case_mae std = 0.0223`
- FFT 可部署替代
  - `fusion_fft_fft_peak_1x_whole__rpm_knn4__tinytcn_all_channels_midband__w0.3`
  - `case_mae mean = 0.1675`
  - `case_mae std = 0.0202`

两者差值：

- `delta_case_mae_mean = +0.0048`
- FFT 替代方案优于 `true_rpm` 上界参考的 seed 数：
  - `4 / 10`

这说明：

- FFT 替代后的融合路线虽然没有严格超过 `true_rpm` 上界；
- 但它和上界之间的平均差距已经缩小到很小的量级；
- 在当前 added 方向上，这已经足以把 FFT 视作“可部署替代”，而不是只能当作弱参考。

### 3.3 [2026-04-07] FFT + learned 融合整体优于 FFT 单独链

- 状态：`current`
- 首次确认：`2026-04-07`
- 最近复核：`2026-04-07`

关键对照：

- FFT 单独 deployable 基线
  - `fft_fft_peak_1x_whole__to__rpm_knn4`
  - `case_mae mean = 0.1860`
- 最优 FFT 融合候选
  - `fusion_fft_fft_peak_1x_whole__rpm_knn4__tinytcn_all_channels_midband__w0.3`
  - `case_mae mean = 0.1675`

两者差值：

- `delta_case_mae_mean = -0.0185`
- FFT 融合优于 FFT 单独链的 seed 数：
  - `8 / 10`

这说明：

- 当前 FFT 支线与 learned midband 仍然呈互补关系，不应把 FFT 看成 learned 分支的替代；
- added 方向当前更合理的 deployable 结构，仍是“解析支线 + learned 中频支线”的组合。

## 4. 当前判断

截至 `2026-04-07`，可以把结论收敛为：

- 若问题限定为 `data/added/` 的 `工况21-24`，FFT 求 RPM 已经可以替代 `true_rpm`，进入最终融合算法的可部署版本；
- 这个“替代”更准确地说是：
  - `true_rpm` 上界的近似 deployable 替代
  - 而不是与 `true_rpm` 完全等价；
- 当前 added 方向更稳的默认 deployable 候选应更新为：
  - `fft_peak_1x_whole -> rpm_knn4 + TinyTCN all_channels midband`
  - 默认固定权重优先参考 `0.3`；
- 但若问题扩大成“全局统一主线是否已经成立”，当前证据仍然不足：
  - 本 try 只覆盖了 added 外部域；
  - 尚未把同一条 `FFT + learned midband` 融合链在 `final LOCO` 上按同口径补齐。

因此，当前更稳妥的工程表达应是：

- FFT 已可替代 `true_rpm` 进入 added deployable 融合线；
- 是否能直接升级成覆盖 `final + added` 的统一默认主线，仍需后续双域复核。
