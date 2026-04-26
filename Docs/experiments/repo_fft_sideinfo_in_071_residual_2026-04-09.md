# repo FFT side-info 并入 071 residual quickcheck（2026-04-09）

- 状态：`current`
- 首次确认：`2026-04-09`
- 最近复核：`2026-04-09`
- 数据范围：
  - `data/added/` 的带标签工况 `21-24`
  - `data/added2/` 的带标签工况 `25-30`
- 代码口径：
  - `src/try/079_repo_fft_sideinfo_in_071_residual/`
- 证据入口：
  - `outputs/try/079_repo_fft_sideinfo_in_071_residual/external_feature_table.csv`
  - `outputs/try/079_repo_fft_sideinfo_in_071_residual/all_case_predictions.csv`
  - `outputs/try/079_repo_fft_sideinfo_in_071_residual/summary_by_protocol.csv`
  - `outputs/try/079_repo_fft_sideinfo_in_071_residual/summary_by_protocol_and_domain.csv`
  - `outputs/try/079_repo_fft_sideinfo_in_071_residual/added_to_added2_compare_vs_071.csv`
  - `outputs/try/079_repo_fft_sideinfo_in_071_residual/summary.md`

## 1. 目标

验证下面这条最小升级版是否优于当前 added-first 默认最佳模型 `071`：

- 当前：
  - `rpm_knn4 + embedding residual ridge`
- 升级：
  - `rpm_knn4 + (embedding + repo_fft side-info) residual ridge`

约束固定为：

- 不改主干；
- 不改模型类型；
- 不加新 gate；
- 主排序口径只看 `added_to_added2`；
- 次级复核口径保留 `added + added2 external LOOCV`。

## 2. 本轮 FFT side-info

只追加下列 FFT 信息：

- `repo_fft_rpm`
- `repo_delta_rpm = repo_fft_rpm - true_rpm`
- `repo_abs_delta_rpm`
- `repo_fft_confidence`
- `repo_fft_source`
  - `fft_peak_1x_whole`
  - `window_peak_1x_conf_8s`

其中：

- FFT RPM 口径复用 `078`；
- `repo_fft_source` 在 residual ridge 输入里展开为两个 one-hot 指示列；
- residual 监督继续复用 `071` 的训练子集内部 `rpm_knn4` 近似 OOF residual。

## 3. 当前结果

### 3.1 [2026-04-09] 最小版没有超过 `071`

`added_to_added2` 主口径结果：

- `071 | rpm_knn4 + embedding residual ridge`
  - `case_mae = 0.6161`
- `079 | rpm_knn4 + embedding + repo_fft side-info residual ridge`
  - `case_mae = 0.7292`
- `rpm_knn4`
  - `case_mae = 1.2903`

`added + added2 external LOOCV` 次口径结果：

- `rpm_knn4`
  - `case_mae = 0.7772`
- `071 | rpm_knn4 + embedding residual ridge`
  - `case_mae = 0.8451`
- `079 | rpm_knn4 + embedding + repo_fft side-info residual ridge`
  - `case_mae = 1.0584`

这说明：

- 最小版虽然仍优于纯 `rpm_knn4`；
- 但它没有超过 `071`；
- 并且在次口径上还出现了更明显退化；
- 因此它不满足当前“升级版”的出线条件。

### 3.2 [2026-04-09] 改善不是稳定性的，而是 `3/6` 分裂

相对 `071` 的 `added_to_added2` case 级对照：

- 变好：
  - `工况27`
    - `abs_error: 0.6505 -> 0.0310`
  - `工况28`
    - `abs_error: 1.0697 -> 0.9087`
  - `工况30`
    - `abs_error: 0.6729 -> 0.1099`
- 变差：
  - `工况25`
    - `abs_error: 1.2068 -> 2.1231`
  - `工况26`
    - `abs_error: 0.0909 -> 0.4272`
  - `工况29`
    - `abs_error: 0.0059 -> 0.7750`

统计上：

- 相对 `071`：
  - `better_case_count = 3 / 6`
  - `worse_case_count = 3 / 6`
- 相对 `rpm_knn4`：
  - `better_case_count = 4 / 6`
  - `worse_case_count = 2 / 6`

这说明：

- FFT side-info 的作用不是“稳定抬升”；
- 更像把 `071` 的 residual 方向重新拽向另一组 case；
- 当前不能把它解释为普适升级。

### 3.3 [2026-04-09] hardest cases 只改善了 `25/27` 中的一个

本轮预期最该受益的是：

- `工况25`
- `工况27`

实际结果：

- `工况27`
  - 从 `0.6505` 改善到 `0.0310`
- `工况25`
  - 从 `1.2068` 恶化到 `2.1231`

这说明：

- FFT side-info 对“RPM/转频结构敏感 case”并非完全无效；
- 但当前信号只落在 `工况27`，没有同时覆盖 `工况25`；
- 因而还不满足“不能只是局部偶然变好”的要求。

### 3.4 [2026-04-09] 恶化与 `window_peak_1x_conf_8s` 主导的低速 case 更相关

`added2` 六个测试工况中：

- 只有 `工况26` 使用 `fft_peak_1x_whole`
- 其余 `工况25 / 27 / 28 / 29 / 30` 都使用 `window_peak_1x_conf_8s`

其中明显恶化的 `工况25 / 29` 都来自：

- `repo_fft_source = window_peak_1x_conf_8s`

并且：

- `工况29`
  - `repo_delta_rpm = -6.0`
  - `071 abs_error = 0.0059`
  - `079 abs_error = 0.7750`
- `工况25`
  - `repo_delta_rpm = +9.2`
  - `071 abs_error = 1.2068`
  - `079 abs_error = 2.1231`

这说明：

- side-info 不是简单提供“更多有用证据”；
- 它同时也把 FFT 估计误差的方向性带进了 residual ridge；
- 在当前只用 `added(21-24)` 四个工况训练的条件下，这个自由度偏大。

## 4. 当前判断

截至 `2026-04-09`，本轮结论应表达为：

- 把 repo FFT side-info 直接并入 `071` residual ridge，当前没有形成可升级的正信号；
- 这条最小版路线没有通过主口径 `0.6161`；
- 改善分布也不满足“不是某一个工况偶然变好”的要求；
- 因此按预定规则，这条线到这里应停止，不再展开 `delta-only` follow-up。

## 5. 一句话版结论

截至 `2026-04-09`，`repo_fft_rpm / delta / confidence / source` 作为 side-info 直接注入 `071` residual ridge，并没有把 added-first 默认最佳模型从 `0.6161` 再往前推；它只在 `工况27` 等少数 case 上显示局部帮助，但整体上不足以构成新的默认升级版。
