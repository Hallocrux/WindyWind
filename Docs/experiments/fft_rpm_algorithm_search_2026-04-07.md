# FFT 解析 RPM 算法搜索（2026-04-07）

- 状态：`current`
- 首次确认：`2026-04-07`
- 最近复核：`2026-04-07`
- 数据范围：
  - `final` 带标签工况 `1, 3-20`
  - `added` 带标签工况 `21-24`
- 代码口径：
  - `src/try/043_1_fft_rpm_algorithm_search/`
- 证据入口：
  - `outputs/try/043_1_fft_rpm_algorithm_search/summary.csv`
  - `outputs/try/043_1_fft_rpm_algorithm_search/case_level_predictions.csv`
  - `outputs/try/043_1_fft_rpm_algorithm_search/summary.md`

## 目标

- 用纯解析方法重建 `sensor -> pred_rpm` 支线；
- 对比整段 FFT、Welch、滑窗 FFT、多谐波模板、峰值回投票、自相关先验等候选；
- 判断哪类 FFT 方向算法最适合作为 `043` 之后的解析 RPM 主候选。

## 默认输入与约束

- 只使用加速度通道 `WSMS*.Acc*`；
- 清洗逻辑复用 `src/current.data_loading`；
- 搜索频带固定为 `1.2Hz - 4.8Hz`，即约 `72 - 288 rpm`；
- 评估指标使用 case 级 `rpm` 绝对误差，不接下游 `wind` 映射。

## 本轮候选算法

- `fft_peak_1x_whole`
- `fft_peak_1x_welch_8s`
- `fft_peak_1x_welch_12s`
- `fft_peak_3x_whole`
- `window_peak_1x_conf_8s`
- `window_peak_1x_conf_12s`
- `fft_vote_whole`
- `harmonic_template_whole`
- `harmonic_template_autocorr_whole`
- `harmonic_template_welch_12s`
- `harmonic_template_autocorr_welch_12s`
- `window_template_conf_12s`
- `window_template_autocorr_conf_12s`
- `window_vote_conf_12s`

## 核心结果

### 1. 当前跨域最优：`hybrid_peak_1x_whole_window8_gate150`

- `all_labeled case_mae = 7.2696`
- `final_direct case_mae = 8.7158`
- `added_external case_mae = 0.4000`

该规则的形式很简单：

- 同时计算整段 `1x peak` 与 `8s` 滑窗 `1x peak`；
- 若两者一致，直接保留；
- 若两者不一致，则：
  - 只要较大候选已经达到 `150 rpm`，保留较大者；
  - 若两者都低于 `150 rpm`，保留较小者。

这说明在当前数据上，`whole` 与 `8s window` 的关系更像互补而不是替代；一个极轻量的双尺度规则已经能同时改善 `final` 与 `added`。

### 2. added 外部域次优但最稳单变体：`fft_peak_1x_whole`

- `case_mae = 0.9500`
- `case_rmse = 1.6279`
- `confidence_mean = 0.4357`

case 级结果：

- `工况21`: `168.0 -> 168.0`
- `工况22`: `106.0 -> 109.2`
- `工况23`: `240.0 -> 240.6`
- `工况24`: `204.0 -> 204.0`

这说明在 added 外部域上，最稳的解析 RPM 候选不是多谐波模板，也不是滑窗聚合，而是“整段加速度频谱上的 `1x` 主峰”。

### 3. 单变体里的全体带标签工况最优：`window_peak_1x_conf_8s`

- `case_mae = 8.8261`
- `case_rmse = 16.2811`
- `confidence_mean = 0.5261`

其中：

- `final_direct`: `case_mae = 10.0000`
- `added_external`: `case_mae = 3.2500`

这说明若只允许保留单一解析变体，`8s` 滑窗 `1x peak` 比整段 `1x peak` 更稳。

### 4. 多谐波模板 / 峰值回投票本轮没有超过简单 `1x peak`

- `harmonic_template_whole`: `all_labeled case_mae = 58.5217`
- `harmonic_template_autocorr_whole`: `all_labeled case_mae = 46.9043`
- `fft_vote_whole`: `all_labeled case_mae = 88.8870`

这说明在当前数据上，谐波模板更容易被半阶 / 倍频混淆带偏；在没有更强消歧规则前，不应默认认为“用更多谐波就一定比单峰更强”。

## 失败模式

`hybrid_peak_1x_whole_window8_gate150` 的 hardest cases 主要集中在：

- `工况1`: `82.0 -> 142.2`
- `工况4`: `172.0 -> 142.2`
- `工况5`: `166.0 -> 142.2`
- `工况3`: `158.0 -> 142.8`
- `工况6`: `195.0 -> 180.0`

这说明当前解析 RPM 仍存在一类“被固定中低频带吸附到约 `2.37Hz` (`142 rpm`)”的失败模式。它不像 `added` 那样是外部域漂移，更像旧域内部若干工况的结构频率 / 机械频率混叠。

## 当前判断

- 若目标是当前默认解析 RPM 候选，优先级应更新为 `hybrid_peak_1x_whole_window8_gate150`；
- 若目标是 added 外部最朴素、最容易解释的单变体，仍可保留 `fft_peak_1x_whole` 作为参考基线；
- 若目标是只允许一个非融合解析变体，`window_peak_1x_conf_8s` 仍是最稳单变体；
- 下一步若继续做 FFT 方向，不应先加更复杂模型，而应先围绕“`142 rpm` 吸附现象”的消歧做规则增强，例如：
  - `whole + 8s window` 多尺度一致性；
  - `1x` 与 `3x` 的一致性检查；
  - 对可疑低频固定峰增加惩罚或旁证。
