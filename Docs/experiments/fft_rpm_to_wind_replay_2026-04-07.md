# FFT RPM -> Wind 结果回放（2026-04-07）

- 状态：`current`
- 首次确认：`2026-04-07`
- 最近复核：`2026-04-07`
- 数据范围：
  - `final` 带标签工况 `1, 3-20`
  - `added` 带标签工况 `21-24`
- 代码口径：
  - `src/try/043_2_fft_rpm_to_wind_replay/`
- 证据入口：
  - `outputs/try/043_2_fft_rpm_to_wind_replay/fft_rpm_to_wind_summary.csv`
  - `outputs/try/043_2_fft_rpm_to_wind_replay/fft_vs_tcn_summary.csv`
  - `outputs/try/043_2_fft_rpm_to_wind_replay/summary.md`

## 目标

- 不重跑 FFT / TinyTCN 主实验；
- 直接复用：
  - `043_1` 的 case 级 FFT RPM 结果；
  - `043` 的 `rpm_knn4 / rpm_linear / ridge_rpm_to_wind` 标量映射口径；
- 验证 `fft_rpm -> wind` 相对 `pred_rpm TinyTCN -> wind` 的可部署收益。

## 默认口径

- `final_loco`：
  - 对每个测试工况，仍按 `043` 口径使用其余 `final` 工况的 `true_rpm / true_wind_speed` 拟合标量映射；
- `added_external`：
  - 使用全部 `final` 带标签工况作为映射训练池；
- 不重复训练任何 `pred_rpm` 模型，只重放标量映射。

## 核心结果

### 1. added 外部最优 deployable 链已切换为 `fft_peak_1x_whole -> rpm_knn4`

- 变体：`fft_fft_peak_1x_whole__to__rpm_knn4`
- `case_mae = 0.1860`
- `case_rmse = 0.2209`

相对 `043` 的最优 TCN deployable 链：

- `pred_rpm_2.0s__to__rpm_linear`
- `case_mae = 1.8886`

当前差值：

- `delta = -1.7026`

这说明只复用现有 FFT RPM 结果，不再重训任何主模型，added 外部可部署链路就已经明显优于原 `pred_rpm TinyTCN` 路线。

### 2. final 旧域内，FFT deployable 链已经接近 TCN deployable 链

- FFT 最优：
  - `fft_window_peak_1x_conf_8s__to__rpm_knn4`
  - `case_mae = 0.4148`
- TCN 最优：
  - `pred_rpm_2.0s__to__rpm_linear`
  - `case_mae = 0.3917`

当前差值：

- `delta = +0.0230`

这说明 FFT 解析支线虽然在 `final` 的 RPM 纯误差上还不如 `TinyTCN@3.0s`，但一旦接上 `rpm -> wind` 标量映射，最终 `wind` 级误差已经和 TCN deployable 链非常接近。

### 3. added 上数值略优于当前 `true_rpm__to__rpm_knn4` 参考，不应过度解读为物理上超过真实 RPM 上界

- `fft_fft_peak_1x_whole__to__rpm_knn4`
  - `case_mae = 0.1860`
- `true_rpm__to__rpm_knn4`
  - `case_mae = 0.2293`

这更合理的解释是：

- 当前 `rpm_knn4` 是小样本、非参数的标量映射；
- added 只有 `4` 个 case；
- 轻微的 RPM 偏差有时会把输入推到对当前 `knn4` 更有利的位置；
- 因而这里的“优于 true_rpm 参考”更像映射偏差或标签稀疏下的数值现象，而不是信息论意义上的真正上界被超过。

## 当前判断

- 若目标是当前 added 可部署默认链，优先级已经可以更新为：
  - `fft_peak_1x_whole -> rpm_knn4`
- 若目标是兼顾 `final + added` 两域，当前更稳的解析 deployable 候选为：
  - `window_peak_1x_conf_8s -> rpm_knn4`
- 下一步若继续迭代，应优先做：
  - `43_1` 中 `142 rpm` 吸附工况的消歧；
  - 然后再重新回放 `rpm -> wind`，而不是回头扩展 TCN `pred_rpm` 训练。
