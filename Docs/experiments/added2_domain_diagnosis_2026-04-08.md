# added2 域诊断（2026-04-08）

- 状态：`current`
- 首次确认：`2026-04-08`
- 最近复核：`2026-04-08`
- 数据范围：
  - `data/added2/` 的 `工况25-30`
  - 参考池：`data/final/` 的带标签工况 `1,3-20`
  - 对照外部域：`data/added/` 的 `工况21-24`
- 代码口径：
  - `src/try/062_added2_domain_diagnosis/`
- 证据入口：
  - `data/added2/dataset_manifest.csv`
  - `outputs/try/062_added2_domain_diagnosis/case_inventory.csv`
  - `outputs/try/062_added2_domain_diagnosis/nearest_reference_cases.csv`
  - `outputs/try/062_added2_domain_diagnosis/added2_domain_summary.csv`
  - `outputs/try/062_added2_domain_diagnosis/added2_feature_outliers_vs_final.csv`
  - `outputs/try/062_added2_domain_diagnosis/label_rpm_consistency.csv`
  - `outputs/try/062_added2_domain_diagnosis/mechanism_projection.png`
  - `outputs/try/062_added2_domain_diagnosis/summary.md`

## 1. 目标

回答三个问题：

1. `added2` 的截图标签与解析 RPM 链是否自洽；
2. `added2` 在 `final + added` 参考池里更像哪类局部子域；
3. 这批新数据更适合直接并入训练，还是更适合先做诊断、route / gate 校准与 reference 扩充。

## 2. 方法口径

- 把 `data/added2/` 的 `6` 个原始 CSV 标准化为：
  - `data/added2/dataset_manifest.csv`
  - `data/added2/standardized_datasets/工况25-30.csv`
- 清洗与切窗继续复用：
  - `src/current.data_loading`
  - `src/current.features`
- 机制特征继续复用：
  - `src/try/030_case_mechanism_clustering/`
- RPM 解析链继续复用：
  - `src/try/043_1_fft_rpm_algorithm_search/` 的 `fft_peak_1x_whole`
  - `window_peak_1x_conf_8s`
  - 同门限的 `hybrid gate150`
- 诊断输出包括：
  - `added2 -> final+added` 最近邻
  - `added2` 相对 `final` 的主异常特征 `z-score`
  - 截图 `rpm` 与 FFT `rpm` 的一致性
  - `label_rpm -> rpm_knn4` 与 `fft_rpm -> rpm_knn4` 的风速误差

## 3. 当前结果

### 3.1 [2026-04-08] `added2` 的 RPM 标签整体自洽，主要不确定性不在转速链

`fft_peak_1x_whole` 相对截图 `rpm` 的误差：

- `工况25`: `7.4 rpm`
- `工况26`: `0.4 rpm`
- `工况27`: `1.4 rpm`
- `工况28`: `2.6 rpm`
- `工况29`: `0.0 rpm`
- `工况30`: `0.4 rpm`

这说明：

- 截至 `2026-04-08`，`added2` 的 `rpm` 标签没有表现出像“明显抄错 / 串号”的迹象；
- 这批新数据的主要矛盾不在转速标签，而更像在风速映射与域位置。

### 3.2 [2026-04-08] `added2` 不是单一新域，而是至少分成三类子域

最近邻结果：

- `工况25 -> 工况20(final) / 工况23(added) / 工况19(final) / 工况13(final)`
- `工况26 -> 工况23(added) / 工况20(final) / 工况13(final) / 工况19(final)`
- `工况27 -> 工况23(added) / 工况20(final) / 工况12(final) / 工况14(final)`
- `工况28 -> 工况4(final) / 工况5(final) / 工况3(final) / 工况9(final)`
- `工况29 -> 工况4(final) / 工况3(final) / 工况5(final) / 工况9(final)`
- `工况30 -> 工况24(added) / 工况19(final) / 工况22(added) / 工况9(final)`

这说明：

- `工况25-27` 更像高转速子域，且与旧 `added` 的 `工况23` 明显接近；
- `工况28-29` 更像低风速、低转速的 `final` 主簇延伸，而不是 added 异常簇；
- `工况30` 同时靠近 `工况24 / 22`，更像新的 added 异常点，而不是普通低风速样本；
- 因此 `added2` 不应被当成一个整块直接并池。

### 3.3 [2026-04-08] `工况25-27` 的风速标签相对现有 `rpm_knn4` 映射明显偏高，而 `工况28-30` 基本贴合

`label_rpm -> rpm_knn4(final)` 的风速绝对误差：

- `工况25`: `1.599`
- `工况26`: `1.595`
- `工况27`: `1.082`
- `工况28`: `0.168`
- `工况29`: `0.272`
- `工况30`: `0.164`

这说明：

- `工况25-27` 在 `rpm` 自洽的前提下，风速标签仍系统性高于现有 `final` 主映射；
- 这更像“高转速端新增了一条更高风速的局部分支”，而不是简单噪声；
- `工况28-30` 对现有 `rpm_knn4` 映射则相对友好，更适合优先拿来做低风险复核与 reference 扩充。

### 3.4 [2026-04-08] `工况30` 再次出现了 added 式的应变低频异常信号

`工况30` 相对 `final` 的主异常特征：

- `strain_low_ratio_median = +4.19 z`
- `strain_mid_ratio_median = -4.07 z`
- `acc_high_ratio_median = -3.76 z`
- `acc_peak_freq_median = +2.81 z`

这说明：

- `工况30` 在机制上并不只是普通低风速样本；
- 它与 `工况22` 一样，带有明显的“应变低频占优 / 中频被压缩”特征；
- 这类样本更适合作为 gate / anomaly canary，而不是默认训练增强样本。

## 4. 当前判断

截至 `2026-04-08`，更合理的利用方式是：

- `工况25-27`
  - 先作为高转速高风速支路的外部验证与标定样本；
  - 用来检查现有 `rpm -> wind` 主映射在高端是否系统性偏低；
- `工况28-29`
  - 优先作为低风险 reference 扩充候选；
  - 可以进入后续的 embedding retrieval / support pool / route 复核；
- `工况30`
  - 优先保留为 added 异常点的 gate / trust 校准样本；
  - 在没有额外约束前，不建议直接并入默认监督训练池。

## 5. 一句话版结论

截至 `2026-04-08`，`added2` 的最大价值不是“再多 6 条数据直接并池”，而是把原本模糊的外部域拆得更清楚：`25-27` 提供高转速高风速的新局部分支，`28-29` 提供接近 `final` 主簇的低风险补样，`30` 则补强了 `工况22` 式低频应变异常的诊断证据；因此更合理的路线是先拿它做诊断、route / gate 与 reference 扩充，再决定是否分支并训。
