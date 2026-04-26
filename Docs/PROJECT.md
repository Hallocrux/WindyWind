# windywind 项目状态卡

本文档只面向 coding agent，记录当前仍有效的项目状态、默认入口、稳定结论和关键约定。  
详细方法、历史结论、模块说明和探索记录统一放到 `Docs/`。

## 1. 项目定位

- 状态：`current`
- 最近复核：`2026-04-05`

这是一个探索性的课程项目，背景材料位于：

- `data/final/00-工程智能基础-工程互联网小项目备课 - 风电5页.pdf`

课程任务主线：

- 任务 1：对每条时程数据做基础统计分析
- 任务 2：基于振动时程与风轮转速预测入流风速
- 任务 3：基于振动时程识别结构基频

## 2. 当前默认入口与仓库结构

- 状态：`current`
- 最近复核：`2026-04-05`

当前默认入口：

- `main.py`
  - 指向 `src/current.pipeline.main`

当前正式代码分层：

- `src/current/`
  - 当前继续迭代的正式版本
- `src/baseline/`
  - 冻结的 baseline 版本
- `src/Baseline_TinyTCN/`
  - `2026-04-05` 固化的 TinyTCN baseline
- `src/try/`
  - 一次性探索目录，不与正式主线混放
- `src/windyWindHowfast/`
  - 视频 RPM 主线
- `src/windNotFound/`
  - 手工标注与 RPM 拟合子项目

## 3. 当前生效的数据组织

- 状态：`current`
- 最近复核：`2026-04-05`
- 数据范围：`工况1` 到 `工况20`

当前表格主线的数据入口由两部分组成：

- `data/final/datasets/工况{ID}.csv`
- `data/final/dataset_manifest.csv`

当前有效规则：

- CSV 文件名只保留工况编号
- 程序不再从文件名解析 `case_id / wind_speed / rpm`
- manifest 是唯一人工元数据来源
- 当前共有 `20` 个工况，其中 `19` 个带标签，`工况2.csv` 无标签

详情见：

- `Docs/data_catalog.md`

## 4. 当前生效的表格主线口径

- 状态：`current`
- 最近复核：`2026-04-05`
- 代码口径：`src/current/`

当前默认清洗口径：

- 删除首尾连续缺失段
- 对中间连续缺失 `<=5` 行的缺失块做线性插值
- 对中间连续缺失 `>5` 行的缺失块直接删除并切分连续段
- 滑动窗口只在连续段内生成
- 忽略 `WSMS00005.*`

当前默认实验口径：

- 按 `5s` 窗长、`2.5s` 步长切窗
- 分组单位为“工况”，不是“窗口”
- 在带标签工况上做 `Leave-One-Condition-Out`
- 无标签工况只做最终推理

详情见：

- `Docs/table_pipeline.md`

## 5. 当前仍有效的稳定结论

### 5.1 [2026-04-05] 当前主数据已切换为 manifest + 标准文件名

- 状态：`current`
- 数据范围：`工况1` 到 `工况20`

当前主线已经停止从文件名解析标签，统一改为：

- `dataset_manifest.csv` 管理人工元数据
- `工况{ID}.csv` 管理物理数据文件

### 5.2 [2026-04-05] 当前宽表主采样率仍稳定在约 50Hz

- 状态：`current`
- 数据范围：`工况1` 到 `工况20`

当前各工况估计采样率仍稳定在约 `50Hz`，默认时间步长继续按 `0.02s` 处理。

### 5.3 [2026-04-05] `WSMS00005.*` 仍应视为错误数据

- 状态：`current`
- 数据范围：当前已知覆盖 `工况5`、`工况12`、`工况13`、`工况14`

当前正式清洗仍必须忽略 `WSMS00005.*`，并把 `WSMS00006.*` 视为有效的第 5 个加速度传感器。

### 5.4 [2026-04-05] 当前表格主线最佳模型为 Ridge + VIB_FT_RPM

- 状态：`current`
- 数据范围：manifest 迁移后的 `工况1` 到 `工况20`
- 代码口径：`src/current/`

当前主流程结果：

- 总窗口数：`866`
- 带标签工况：`19`
- 全局最优模型：`Ridge + VIB_FT_RPM`
- 最佳 rpm-free 模型：`Ridge + VIB_FT`
- `工况2.csv` 当前预测风速：`3.3279 m/s`

### 5.5 [2026-04-05] 新增工况 15-17 的缺失水平显著高于旧工况

- 状态：`current`
- 数据范围：`工况1` 到 `工况20`

当前质量复核显示：

- 平均缺失率：`2.0493%`
- 平均首尾连续缺失删除比例：`4.0543%`
- 缺失率最高：`工况15`，`6.1709%`
- 最长连续缺失段：`工况16`，`352` 点
- 当前保留窗口中的受缺失影响窗口占比：`0.0000%`

这说明新增工况值得继续单独做阶段性质量复核。

详情见：

- `Docs/data_quality_and_findings.md`

### 5.6 [2026-04-05] 当前 exploratory 口径下的最优模型已切换为 TinyTCN

- 状态：`current`
- 数据范围：manifest 迁移后的 `工况1` 到 `工况20`
- 代码口径：
  - `src/try/013_phase3_cnn_tcn_smoke/`
  - `src/Baseline_TinyTCN/`

当前 full 复核结果：

- `TinyTCN`
  - `case_mae = 0.3030`
  - `case_rmse = 0.4930`
- `Tiny1DCNN`
  - `case_mae = 0.3340`
  - `case_rmse = 0.5151`
- `TabularReference_G6_Ridge`
  - `case_mae = 0.4045`
  - `case_rmse = 0.7106`

这说明当前最值得继续迭代的主线候选已经从表格 `Ridge` 候选切换为 `TinyTCN`。

### 5.7 [2026-04-05] 结构基频方向的首个主候选频带已落在约 2.2Hz - 2.4Hz

- 状态：`current`
- 数据范围：`工况1` 到 `工况20`
- 代码口径：
  - `src/current.data_loading`
  - `src/try/018_structural_fundamental_frequency_scan/`

当前去转频候选扫描结果显示：

- 在 `0.5Hz - 8.0Hz` 搜索范围内，屏蔽 `1x - 4x` 转频附近 `±0.2Hz` 后
- `20` 个工况中有 `12` 个工况的应变侧候选频率中位数集中在约 `2.2Hz - 2.4Hz`
- 剩余工况存在约 `2.8Hz - 3.6Hz` 的次候选带
- 结构基频这条线当前应优先继续做“候选频率提取 + 稳定性验证”，而不是直接做监督学习

### 5.8 [2026-04-06] 当前默认样例视频 `VID_20260330_162635.mp4` 对应 `工况5` 实录

- 状态：`current`
- 数据范围：
  - `data/video/VID_20260330_162635.mp4`
  - `data/final/datasets/工况5.csv`
- 证据入口：
  - `config/test.yaml`
  - `outputs/annotations/test/summary.json`
  - `outputs/annotations/test/video_rpm_eval.json`
  - `outputs/dataset_inventory.csv`

当前视频子项目口径按以下事实对齐：

- `VID_20260330_162635.mp4` 是 `工况5` 的实录视频；
- `src/windyWindHowfast/` 是当前 `工况5` 视频的 RPM CV 主线；
- `src/windNotFound/` 是当前 `工况5` 视频的手工标注、RPM 拟合和视频侧验证主线。

### 5.9 [2026-04-06] 当前 TinyTCN rpm 细窗长 full 最优点落在 `3.0s`

- 状态：`current`
- 数据范围：manifest 迁移后的 `工况1` 到 `工况20`
- 代码口径：
  - `src/try/024_tinytcn_rpm_fine_window_scan/`

当前 `2s-5s` 细窗长 full 扫描结果显示：

- `TinyTCN @ 3.0s`
  - `case_mae = 5.1863`
  - `case_rmse = 8.5262`
- `TinyTCN @ 2.0s`
  - `case_mae = 7.8917`
- `TinyTCN @ 2.5s`
  - `case_mae = 8.0277`
- `TinyTCN @ 5.0s`
  - `case_mae = 8.3832`

这说明当前证据更支持“存在中间最优窗长”，而不是“窗口越短越好”。

### 5.10 [2026-04-07] added 外部失配当前主要由应变侧域偏移驱动

- 状态：`current`
- 数据范围：
  - `data/added/` 的 `工况21-24`
- 代码口径：
  - `src/try/037_case22_label_and_modality_check/`

当前 added 模态外部对照结果显示：

- `rpm_knn4`
  - `case_mae = 0.2293`
- `TinyTCN | full_final_pool | acc_only`
  - `case_mae = 0.3553`
- `TinyTCN | clean_final_pool | acc_only`
  - `case_mae = 0.4362`
- `TinyTCN | clean_final_pool | all_channels`
  - `case_mae = 2.0341`
- `TinyTCN | full_final_pool | strain_only`
  - `case_mae = 2.3854`
- `TinyTCN | full_final_pool | all_channels`
  - `case_mae = 3.2158`

这说明：

- 当前 added 外部高估主要由应变侧输入驱动；
- 加速度侧单独建模比“全通道直接外推”稳定得多；
- added 这条线的下一步应优先考虑应变侧域适配与标签证据复核。

### 5.11 [2026-04-07] added 方向上应变高通可显著缓解原始全通道失配

- 状态：`current`
- 数据范围：
  - `data/added/` 的 `工况21-24`
- 代码口径：
  - `src/try/038_strain_shift_mitigation_check/`

当前 added 应变修复快速验证结果显示：

- `rpm_knn4`
  - `case_mae = 0.2293`
- `TinyTCN | full_final_pool | acc_only`
  - `case_mae = 0.3471`
- `TinyTCN | full_final_pool | all_channels_raw`
  - `case_mae = 3.1588`
- `TinyTCN | full_final_pool | all_channels + strain_highpass_2hz`
  - `case_mae = 0.4295`
- `TinyTCN | full_final_pool | all_channels + strain_case_zscore`
  - `case_mae = 0.4563`

这说明：

- 对应变做 `>2Hz` 高通后，added 外推可以从“明显崩坏”恢复到“可参考”；
- 当前 added 方向的主要矛盾确实集中在应变侧低频部分；
- added 这条线后续继续做应变探索时，应优先围绕高通或中频带，而不是恢复到原始全频输入。

### 5.12 [2026-04-07] added 方向上当前最可迁移的应变频带已收敛到约 `3.0-6.0Hz`

- 状态：`current`
- 数据范围：
  - `data/added/` 的 `工况21-24`
- 代码口径：
  - `src/try/040_midband_strain_weight_scan/`

当前应变频带细扫结果显示：

- `TinyTCN | full_final_pool | strain_bandpass_3.0_6.0hz`
  - `case_mae = 0.2584`
  - `case22_abs_error = 0.1324`
- `TinyTCN | full_final_pool | strain_bandpass_3.0_5.0hz`
  - `case_mae = 0.2743`
- `TinyTCN | full_final_pool | acc_only`
  - `case_mae = 0.4044`
- `acc_only + strain_bandpass_3.0_6.0hz`
  - `case_mae = 0.2468`
- `rpm_knn4`
  - `case_mae = 0.2293`

这说明：

- added 方向上，当前最有保留价值的应变信息已从粗粒度 `3-6Hz` 收敛到更精确的约 `3.0-6.0Hz`；
- `3.0-6.0Hz` 中频应变已经优于单独 `acc_only`；
- 这说明后续继续做 added learned 分支时，应优先围绕 `3.0-6.0Hz` 展开，而不是回到更宽频带。

### 5.13 [2026-04-07] added 方向当前更稳的 `true_rpm` 上界候选为 `rpm_knn4 + learned midband @ learned_weight≈0.3`

- 状态：`current`
- 数据范围：
  - `data/added/` 的 `工况21-24`
- 代码口径：
  - `src/try/041_rpm_vs_learned_midband_check/`
  - `src/try/042_rpm_learned_midband_multiseed_stability_check/`

当前 added 单次复核 + 多随机种子复核结果显示：

- `rpm_knn4`
  - `case_mae = 0.2293`
- `rpm_knn4 + TinyTCN all_channels midband @ learned_weight=0.3`
  - `case_mae mean = 0.1627`
  - `case_mae std = 0.0223`
  - `better_than_rpm_rate = 1.0`
- `rpm_knn4 + TinyTCN all_channels midband @ learned_weight=0.5`
  - `case_mae mean = 0.1822`
  - `case_mae std = 0.0341`
  - `better_than_rpm_rate = 0.9`
  - `case22_abs_error mean = 0.0885`

这说明：

- 当前 added 方向在可利用真实 `rpm` 的上界参考里，最稳路线已经确定为“解析基线 + learned 中频分支”的混合方案；
- `rpm_knn4` 与 `learned midband` 的关系更像互补，而不是替代；
- 若目标是 `工况21-24` 的整体平均误差，当前默认固定权重应优先参考更稳的 `0.3`；
- 若目标是尽量修平 `工况22` 这类 hardest case，`0.5` 仍可作为参考权重保留。

### 5.14 [2026-04-07] `pred_rpm` 解析支线当前只在 `final` 域内可用，added 外部域不可直接进入默认候选矩阵

- 状态：`current`
- 数据范围：
  - `final` 带标签工况
  - `data/added/` 的 `工况21-24`
- 代码口径：
  - `src/try/043_pred_rpm_deployability_check/`

当前可部署 `pred_rpm -> wind` 验证结果显示：

- `final LOCO`
  - 最优可部署链：`pred_rpm_3.0s -> ridge_rpm_to_wind`
  - `case_mae = 0.3262`
- `added 21-24`
  - 最优可部署链：`pred_rpm_2.0s -> rpm_linear`
  - `case_mae = 1.8886`
- `added 21-24` 的 `pred_rpm` case 级误差：
  - `2.0s rpm_mae = 74.6552`
  - `3.0s rpm_mae = 77.5826`
  - `5.0s rpm_mae = 93.2017`

这说明：

- `pred_rpm` 支线在旧域内并未失效；
- 但在 added 外部域上出现了明显高估崩坏；
- 在 `pred_rpm` 外部泛化修复前，`C = pred_rpm` 当前只能保留为研究支线，不应进入后续默认融合候选矩阵。

### 5.15 [2026-04-07] FFT 解析 RPM 支线当前最优候选已更新为 `whole 1x + 8s window 1x` 的轻量融合规则

- 状态：`current`
- 数据范围：
  - `final` 带标签工况 `1, 3-20`
  - `data/added/` 的 `工况21-24`
- 代码口径：
  - `src/try/043_1_fft_rpm_algorithm_search/`

当前 FFT 解析 RPM 搜索结果显示：

- `added_external` 最优：
  - `hybrid_peak_1x_whole_window8_gate150`
  - `case_mae = 0.4000`
- `all_labeled` 最优：
  - `hybrid_peak_1x_whole_window8_gate150`
  - `case_mae = 7.2696`
- `final_direct` 最优：
  - `hybrid_peak_1x_whole_window8_gate150`
  - `case_mae = 8.7158`

这说明：

- added 外部域当前最稳的解析 RPM 候选已经不是 learned `pred_rpm`，而是 FFT 方向的轻量双尺度规则；
- 单独看解析单变体时，`8s` 滑窗 `1x peak` 仍优于整段 `1x peak`；
- 但把整段 `1x peak` 与 `8s` 滑窗 `1x peak` 做轻量规则融合后，可以同时改善 `final` 与 `added`；
- FFT 方向下一步的主要问题不再是“有没有转频信息”，而是如何继续消除若干工况被固定中低频带吸附到约 `142 rpm` 的失败模式。

### 5.16 [2026-04-07] 复用 FFT RPM 结果后，added 可部署 `rpm -> wind` 默认链已切换为 `fft_peak_1x_whole -> rpm_knn4`

- 状态：`current`
- 数据范围：
  - `final` 带标签工况 `1, 3-20`
  - `data/added/` 的 `工况21-24`
- 代码口径：
  - `src/try/043_2_fft_rpm_to_wind_replay/`

当前结果回放验证显示：

- `added_external` 最优 FFT deployable 链：
  - `fft_fft_peak_1x_whole__to__rpm_knn4`
  - `case_mae = 0.1860`
- `final_loco` 最优 FFT deployable 链：
  - `fft_window_peak_1x_conf_8s__to__rpm_knn4`
  - `case_mae = 0.4148`
- `043` 中的最优 TCN deployable 链：
  - `added_external`
    - `pred_rpm_2.0s__to__rpm_linear`
    - `case_mae = 1.8886`
  - `final_loco`
    - `pred_rpm_2.0s__to__rpm_linear`
    - `case_mae = 0.3917`

这说明：

- 在 added 外部域上，复用 FFT RPM 的 deployable 链已经明显优于此前 `pred_rpm TinyTCN` 路线；
- 在 final 旧域内，FFT deployable 链虽然仍略弱于最佳 TCN deployable 链，但已经接近同一量级；
- 当前若目标是 added 的可部署默认链，应优先保留 `fft_peak_1x_whole -> rpm_knn4`，而不是继续依赖旧的 TCN `pred_rpm`。

### 5.17 [2026-04-07] FFT 求 RPM 当前已可替代 `true_rpm`，进入 added 方向的可部署融合线

- 状态：`current`
- 数据范围：
  - `data/added/` 的 `工况21-24`
- 代码口径：
  - `src/try/042_rpm_learned_midband_multiseed_stability_check/`
  - `src/try/043_2_fft_rpm_to_wind_replay/`
  - `src/try/043_3_fft_midband_fusion_replay/`

当前 added 结果回放融合显示：

- `true_rpm` 上界参考
  - `rpm_knn4 + TinyTCN all_channels midband @ learned_weight=0.3`
  - `case_mae mean = 0.1627`
- 最优 deployable 替代
  - `fft_peak_1x_whole -> rpm_knn4 + TinyTCN all_channels midband @ learned_weight=0.3`
  - `case_mae mean = 0.1675`
  - `case_mae std = 0.0202`
- 相对 FFT 单独 deployable 基线
  - `fft_peak_1x_whole -> rpm_knn4`
  - `case_mae = 0.1860`

这说明：

- 在 added 方向上，FFT 求 RPM 已经可以作为 `true_rpm` 的近似 deployable 替代，进入最终融合算法；
- 这个替代当前更适合理解为“接近 `true_rpm` 上界”，而不是“与 `true_rpm` 完全等价”；
- 当前 added 方向更稳的默认 deployable 融合候选应优先参考：
  - `fft_peak_1x_whole -> rpm_knn4 + TinyTCN all_channels midband @ learned_weight≈0.3`
- 但该结论当前只覆盖 added 外部域，是否能直接升级为 `final + added` 的统一默认主线，仍需后续双域复核。

### 5.18 [2026-04-08] `TinyTCN embedding` 当前更适合作为 added 局部修正信号，而不是统一主干

- 状态：`current`
- 数据范围：
  - `final` 带标签工况 `1, 3-20`
  - `added` 带标签工况 `21-24`
- 代码口径：
  - `src/try/051_tcn_embedding_knn_residual/`

当前 unified `LOOCV` quickcheck 结果显示：

- `all_labeled | rpm_knn4`
  - `case_mae = 0.3634`
- `all_labeled | rpm_knn4 + embedding_residual_knn4 @ w=0.5`
  - `case_mae = 0.3700`
- `added | rpm_knn4`
  - `case_mae = 0.2293`
- `added | rpm_knn4 + embedding_residual_knn4 @ w=0.5`
  - `case_mae = 0.1698`
- `embedding_knn4`
  - `all_labeled case_mae = 0.5791`

这说明：

- 当前 `TinyTCN` 的 case embedding 确实包含 added 方向的局部修正信号；
- 但这类 embedding 还不能直接替代 `rpm_knn4` 成为 unified 默认主干；
- 当前更合理的角色是：
  - `rpm_knn4` 继续做主干
  - `embedding residual` 作为可能需要门控的 optional correction

### 5.19 [2026-04-08] 代表性 holdout quickcheck 更支持 `2s+8s` 多尺度 embedding residual，而不是单独切到 `5s`

- 状态：`current`
- 数据范围：
  - `final` 代表性 holdout：`工况1 / 3 / 17 / 18`
  - `added`：`工况21-24`
- 代码口径：
  - `src/try/052_tcn_embedding_window_signal_quickcheck/`

当前代表性 holdout quickcheck 结果显示：

- `focus_all | rpm_knn4`
  - `case_mae = 0.6444`
- `focus_all | rpm_knn4 + residual_2s @ w=0.5`
  - `case_mae = 0.6028`
- `focus_all | rpm_knn4 + residual_5s @ w=0.5`
  - `case_mae = 0.6070`
- `focus_all | rpm_knn4 + residual_concat_2s_8s @ w=0.5`
  - `case_mae = 0.6006`

这说明：

- 在当前 residual 方向上，把 `2s` 直接换成 `5s` 没有显示出更强信号；
- 当前更值得继续追的是多尺度 `2s+8s` embedding correction；
- 但该结论当前只覆盖代表性小规模 holdout，不应直接等同于全量 unified `LOOCV` 结论。

### 5.20 [2026-04-08] 当前最小版 support-window residual 尚未超过 case-level `2s+8s` concat residual

- 状态：`current`
- 数据范围：
  - `final` 代表性 holdout：`工况1 / 3 / 17 / 18`
  - `added`：`工况21-24`
- 代码口径：
  - `src/try/053_support_window_residual_quickcheck/`

当前代表性 holdout quickcheck 结果显示：

- `052 | rpm_knn4 + embedding_residual_concat_2s_8s @ w=0.5`
  - `focus_all case_mae = 0.6006`
- `053 | rpm_knn4 + support_window_residual_2s @ w=0.5`
  - `focus_all case_mae = 0.6027`
- `053 | rpm_knn4 + support_window_residual_avg_2s_8s @ w=0.5`
  - `focus_all case_mae = 0.6125`
- `053 | rpm_knn4 + support_window_residual_concat_2s_8s @ w=0.5`
  - `focus_all case_mae = 0.6629`

这说明：

- “先找参考工况，再用参考工况窗口时序做 support” 当前不是完全无效；
- 但当前最小版实现整体上还没有优于更简单的 case-level `2s+8s` concat residual；
- support-window 方向当前更像 added 局部强修复分支，而不是默认主候选。

### 5.21 [2026-04-08] 当前 case-level evidence aggregator quickcheck 仍未超过固定 `2s+8s` concat residual

- 状态：`current`
- 数据范围：
  - `final` 代表性 holdout：`工况1 / 3 / 17 / 18`
  - `added`：`工况21-24`
- 代码口径：
  - `src/try/054_case_evidence_aggregator_quickcheck/`

当前代表性 holdout quickcheck 结果显示：

- `052 | rpm_knn4 + embedding_residual_concat_2s_8s @ w=0.5`
  - `focus_all case_mae = 0.6006`
- `054 | ridge_candidate_only @ w=0.5`
  - `focus_all case_mae = 0.7477`
- `054 | ridge_candidate_plus_mechanism @ w=0.5`
  - `focus_all case_mae = 0.9150`

这说明：

- 把已有 correction 候选再交给一个 case-level 小模型做连续 residual 回归，当前并没有额外增益；
- 在当前极小样本下，这条线更适合作为分析工具，而不是主候选；
- 若继续追 case-level aggregation，更值得改成更保守的 gate / bucket，而不是继续做连续 residual 回归。

### 5.22 [2026-04-08] 当前“先做局部机制路由”只有弱正信号，aligned head 仍未超过固定 `2s+8s` concat residual

- 状态：`current`
- 数据范围：
  - `final` 代表性 holdout：`工况1 / 3 / 17 / 18`
  - `added`：`工况21-24`
- 代码口径：
  - `src/try/055_local_mechanism_router_aligned_head_quickcheck/`

当前代表性 holdout quickcheck 结果显示：

- `052 | rpm_knn4 + embedding_residual_concat_2s_8s @ w=0.5`
  - `focus_all case_mae = 0.6006`
- `055 | rpm_knn4 + local_mechanism_residual_knn4_concat_2s_8s @ w=0.5`
  - `focus_all case_mae = 0.6252`
  - `final_focus case_mae = 1.0064`
- `055 | rpm_knn4 + local_mechanism_aligned_tanh_ridge_pca6 @ w=0.5`
  - `focus_all case_mae = 0.7146`
- `055 | rpm_knn4 + local_mechanism_aligned_tanh_ridge_pca6`
  - `focus_all case_mae = 0.8527`

这说明：

- “先限制到局部相似机制区域”这一步不是完全无效，局部机制收窄对部分 `final` hard case 有弱正信号；
- 但当前机制路由对 added 外部样本还不够稳，`工况22` 的 mechanism distance 已经出现明显失真；
- 在机制池本身还不稳的前提下，继续放大小头并不能自然超过固定 `2s+8s` concat residual；
- 这条线下一步更值得优先修的是机制空间校准与局部路由，而不是继续扩大 aligned head。

### 5.23 [2026-04-08] 去掉 mechanism pool 后，当前最小版 embedding-prototype head 仍未超过固定 `2s+8s` concat residual

- 状态：`current`
- 数据范围：
  - `final` 代表性 holdout：`工况1 / 3 / 17 / 18`
  - `added`：`工况21-24`
- 代码口径：
  - `src/try/056_embedding_topk_local_prototype_fusion/`

当前去掉 mechanism pool 后的高维 prototype quickcheck 结果显示：

- `052 | rpm_knn4 + embedding_residual_concat_2s_8s @ w=0.5`
  - `focus_all case_mae = 0.6006`
- `056 | rpm_knn4 + embedding_topk_prototype_ridge @ w=0.5`
  - `focus_all case_mae = 0.7193`
- `056 | rpm_knn4 + embedding_topk_prototype_ridge`
  - `focus_all case_mae = 0.8526`
- `056` 的局部代表工况：
  - `工况3`
    - `abs_error = 0.7208`
  - `工况17`
    - `abs_error = 1.1421`

这说明：

- 参考池切回 `embedding top-k` 后，方向已经与原始设想对齐；
- “local reference set -> prototype -> constrained correction” 这条线在部分 `final` hard case 上有正信号；
- 但当前最小版高维 `ridge` 头对 added 外部域还不够稳，整体上仍未超过固定 `2s+8s` concat residual；
- 这条线下一步更值得优先修的是 correction 约束与表征压缩，而不是重新回到 mechanism pool。

### 5.24 [2026-04-08] 统一坐标系下的 `embedding_concat` 已表现出可检索的局部结构，可保留为默认检索空间候选

- 状态：`current`
- 数据范围：
  - `final`：`工况1-20`
  - `added`：`工况21-24`
- 代码口径：
  - `src/try/057_embedding_space_diagnosis/`

当前统一 embedding 空间诊断结果显示：

- PCA explained variance：
  - `PC1 = 62.18%`
  - `PC2 = 25.59%`
- overall same-domain neighbor rate：`83.33%`
- `final_labeled` same-domain neighbor rate：`94.74%`
- `added` same-domain neighbor rate：`50.00%`
- `added` 邻域摘要：
  - `工况21 -> 24 / 22 / 7 / 10`
  - `工况22 -> 21 / 24 / 1 / 3`
  - `工况23 -> 16 / 15 / 17 / 24`
  - `工况24 -> 21 / 22 / 10 / 23`

这说明：

- `embedding_concat` 在统一坐标系下已经形成明确的局部结构；
- 它比 `mechanism pool` 更适合作为默认候选检索空间；
- `added` 并非完全断裂，而是形成了约两组局部邻域；
- 但 `added` 的局部域还没有完全闭合，后续继续用它做 prototype / correction 时仍需要保守约束和稳定性复核。

### 5.25 [2026-04-08] prototype head 的 added 退化当前主要来自 full input 自由度过大，`delta-only` 明显更稳

- 状态：`current`
- 数据范围：
  - `final` 代表性 holdout：`工况1 / 3 / 17 / 18`
  - `added`：`工况21-24`
- 代码口径：
  - `src/try/058_prototype_head_ablation_quickcheck/`

当前 prototype head 最小版 ablation 结果显示：

- `056 | embedding_topk_prototype_ridge @ w=0.5`
  - `added_focus case_mae = 0.3514`
- `058 | delta_only_prototype_ridge @ w=0.5`
  - `added_focus case_mae = 0.2612`
- `058 | lowrank_delta_prototype_ridge @ w=0.5`
  - `added_focus case_mae = 0.3186`
- 对照：
  - `rpm_knn4`
    - `added_focus case_mae = 0.2293`

这说明：

- `056` 的 added 退化主要确实来自 full input head 自由度过大；
- 把 correction 收紧到 `delta-only` 后，added 退化已明显缓解；
- 但单纯做低维 `delta PCA` 当前还没有比 `delta-only` 更稳；
- 这支持后续继续沿“embedding top-k 检索 + 更保守的 delta correction / gate”推进，而不是回到 full input prototype head。

### 5.26 [2026-04-08] 在 `delta-only` 上叠加保守 `trigger` 后，当前已能把 final 额外伤害压回 `rpm_knn4` 附近，并保留部分 added 收益

- 状态：`current`
- 数据范围：
  - `final` 代表性 holdout：`工况1 / 3 / 17 / 18`
  - `added`：`工况21-24`
- 代码口径：
  - `src/try/059_delta_only_gate_bucket_trigger_quickcheck/`

当前 `delta-only` 路由 quickcheck 结果显示：

- `rpm_knn4`
  - `final_focus case_mae = 1.0595`
  - `added_focus case_mae = 0.2293`
- `058 | rpm_knn4 + delta_only_prototype_ridge @ w=0.5`
  - `final_focus case_mae = 1.1828`
  - `added_focus case_mae = 0.2612`
- `059 | delta_only_trigger_rule_cv`
  - `final_focus case_mae = 1.0595`
  - `added_focus case_mae = 0.1932`
  - `focus_all case_mae = 0.6264`
- `059 | delta_only_two_stage_hgb_t0.65`
  - `added_focus case_mae = 0.1926`
  - `final_focus case_mae = 1.2024`
- 对照：
  - `052 | rpm_knn4 + embedding_residual_concat_2s_8s @ w=0.5`
  - `focus_all case_mae = 0.6006`

这说明：

- `delta-only` 继续适合作为默认 correction 头；
- 但它更适合先经过保守 `trigger`，而不是直接做连续 `soft gate` 或激进 `bucket`；
- `trigger` 当前已经能把 final 侧的额外伤害压回 `rpm_knn4` 基线附近，同时保留一部分 added 改善；
- `bucket / two-stage` 虽然能保留 added 收益，但在当前小样本 quickcheck 下仍会重新放大 final 伤害；
- 这条线下一步更值得优先继续修“何时触发 correction”，而不是重新放大 correction 强度；
- 但当前证据仍未超过固定 `2s+8s` concat residual，不应直接替代 `052` 的默认参考位。

### 5.27 [2026-04-08] `prototype alignment` 当前已在 final hard case 上显示独立正信号，但 added 外部域仍需额外 trust gate

- 状态：`current`
- 数据范围：
  - `final` 代表性 holdout：`工况1 / 3 / 17 / 18`
  - `added`：`工况21-24`
- 代码口径：
  - `src/try/060_embedding_topk_prototype_alignment_quickcheck/`

当前 `prototype alignment` quickcheck 结果显示：

- `rpm_knn4`
  - `final_focus case_mae = 1.0595`
  - `added_focus case_mae = 0.2293`
  - `focus_all case_mae = 0.6444`
- `060 | embedding_prototype_alignment_ridge`
  - `final_focus case_mae = 0.7951`
  - `added_focus case_mae = 0.8009`
- `060 | embedding_prototype_alignment_ridge @ w=0.5`
  - `final_focus case_mae = 0.9262`
  - `added_focus case_mae = 0.3537`
  - `focus_all case_mae = 0.6399`
- 对照：
  - `052 | rpm_knn4 + embedding_residual_concat_2s_8s @ w=0.5`
  - `focus_all case_mae = 0.6006`

这说明：

- “先构造局部 prototype，再在表征层做 alignment，再做 bounded prediction” 这条新融合定义本身是成立的；
- 它在 `final` hard case 上已经显示出比 `rpm_knn4` 更强的独立正信号；
- 当前主要短板不在 alignment 思路本身，而在 added 外部域的 prototype 可信度不足；
- 因此这条线下一步更值得继续加的是 `trust gate / domain gate`，而不是退回到邻居 residual 平均；
- 但在 added 约束补齐前，它还不适合直接升级为 unified 默认主候选。

### 5.28 [2026-04-08] 把 `prototype alignment` 放到 `full final LOCO` 与 `added` 上界口径后，当前版仍未达到两侧 SOTA

- 状态：`current`
- 数据范围：
  - `final`：带标签工况全量 `LOCO`
  - `added`：`工况21-24`
- 代码口径：
  - `src/try/061_prototype_alignment_sota_recheck/`

当前同口径 SOTA 复核结果显示：

- `final SOTA`
  - `TinyTCN_multiscale_late_fusion_2s_8s`
  - `case_mae = 0.1685`
- `061 | rpm_knn4`
  - `final_loco case_mae = 0.4256`
- `061 | embedding_prototype_alignment_ridge @ w=0.5`
  - `final_loco case_mae = 0.4904`
- `added 上界 SOTA`
  - `rpm_knn4 + TinyTCN all_channels midband @ w=0.3`
  - `case_mae mean = 0.1627`
- `061 | rpm_knn4`
  - `added_external case_mae = 0.2293`
- `061 | embedding_prototype_alignment_ridge @ w=0.5`
  - `added_external case_mae = 0.4410`

这说明：

- `060` 在代表性 hard-case quickcheck 上出现的正信号，并没有迁移成同口径 SOTA 结果；
- 放到 `full final LOCO` 后，当前版还没有超过 `rpm_knn4`；
- 放到 `added 21-24` 后，当前版与 added 上界 SOTA 的差距更明显；
- 因此到这一步，更合理的结论是：
  - `prototype alignment` 仍可保留为分析型表征方向；
  - 但“当前版实现”不应继续作为冲击 unified 主线 SOTA 的优先候选；
  - 若后续继续追这条线，必须先明确新的约束机制，而不是把当前版直接放大复训。

### 5.29 [2026-04-08] `final` 训练的 `TinyTCN 2s+8s` 晚融合当前不能直接外推到 `added2`

- 状态：`current`
- 数据范围：
  - 训练域：`final` 带标签工况 `1, 3-20`
  - 外部域：`data/added2/` 的 `工况25-30`
- 代码口径：
  - `src/try/063_final_late_fusion_added2_replay/`
- 模型资产：
  - `outputs/try/063_final_late_fusion_added2_replay/models/checkpoints/final_deploy_2s.pt`
  - `outputs/try/063_final_late_fusion_added2_replay/models/checkpoints/final_deploy_2s_norm.npz`
  - `outputs/try/063_final_late_fusion_added2_replay/models/checkpoints/final_deploy_2s.json`
  - `outputs/try/063_final_late_fusion_added2_replay/models/checkpoints/final_deploy_8s.pt`
  - `outputs/try/063_final_late_fusion_added2_replay/models/checkpoints/final_deploy_8s_norm.npz`
  - `outputs/try/063_final_late_fusion_added2_replay/models/checkpoints/final_deploy_8s.json`

当前 `full final deploy` 回放结果显示：

- `added2 | TinyTCN@2s`
  - `case_mae = 2.1985`
- `added2 | TinyTCN@8s`
  - `case_mae = 2.4157`
- `added2 | TinyTCN 2s+8s late fusion`
  - `case_mae = 2.2897`
- `added2 | TinyTCN 2s+8s late fusion`
  - `mean_signed_error = +2.1177`

这说明：

- `026` 在 `final` 域里成立的 `2s+8s` 优势，没有自然迁移到 `added2`；
- 在 `added2` 上，direct learned 路线仍以系统性高估为主；
- 当前若目标是 `added2` 外推，`2s+8s` 不应直接作为默认 direct learned 候选。

### 5.30 [2026-04-08] `added` 上界 SOTA 的 midband 融合当前也不能直接迁移到 `added2`

- 状态：`current`
- 数据范围：
  - 训练域：`final` 带标签工况 `1, 3-20`
  - 旧外部域：`data/added/` 的 `工况21-24`
  - 新外部域：`data/added2/` 的 `工况25-30`
- 代码口径：
  - `src/try/064_added_sota_added2_replay/`
  - 为同时评估 `added2`，信号列取 `final / added / added2` 的共同列；模型结构、`3.0-6.0Hz` 频带、`5s` 窗长和 seed 列表复用 `042`
- 证据入口：
  - `outputs/try/064_added_sota_added2_replay/comparison_overview.csv`
  - `outputs/try/064_added_sota_added2_replay/stability_overview_by_domain.csv`
  - `outputs/try/064_added_sota_added2_replay/added2_case_comparison.csv`
  - `outputs/try/064_added_sota_added2_replay/summary.md`
- 模型资产：
  - `outputs/try/064_added_sota_added2_replay/model_assets.csv`
  - `outputs/try/064_added_sota_added2_replay/models/checkpoints/final_deploy_midband_seed*_w5s.pt`
  - `outputs/try/064_added_sota_added2_replay/models/checkpoints/final_deploy_midband_seed*_w5s_norm.npz`
  - `outputs/try/064_added_sota_added2_replay/models/checkpoints/final_deploy_midband_seed*_w5s.json`
  - `outputs/try/063_final_late_fusion_added2_replay/models/checkpoints/final_deploy_2s.pt`
  - `outputs/try/063_final_late_fusion_added2_replay/models/checkpoints/final_deploy_8s.pt`

当前同口径回放显示：

- `added | rpm_knn4 + TinyTCN all_channels midband @ w=0.3`
  - `case_mae mean = 0.1596`
- `added | rpm_knn4`
  - `case_mae mean = 0.2293`
- `added2 | rpm_knn4`
  - `case_mae mean = 0.8131`
- `added2 | rpm_knn4 + TinyTCN all_channels midband @ w=0.3`
  - `case_mae mean = 1.0126`
- `added2 | direct TinyTCN 2s+8s late fusion`
  - `case_mae = 2.2897`

这说明：

- `midband` 在旧 `added` 上的收益已经被 `064` 复核到，但没有迁移到 `added2`；
- `added2` 上整体更稳的默认候选仍是 `rpm_knn4`；
- `direct 2s+8s` 只在 `工况25-26` 有局部优势，在 `工况28-30` 明显失效；
- 若要利用 `added2`，下一步更适合做 case-level route / gate，而不是把 `2s+8s` 或 `midband` 直接全局开启。

### 5.31 [2026-04-08] 若必须给出统一稳健的最终风速回归主干，当前应优先参考 `rpm_knn4`

- 状态：`current`
- 数据范围：
  - `final` 带标签工况 `1, 3-20`
  - `added` 带标签工况 `21-24`
  - `added2` 带标签工况 `25-30`
- 代码口径：
  - `src/try/037_case22_label_and_modality_check/`
  - `src/try/042_rpm_learned_midband_multiseed_stability_check/`
  - `src/try/063_final_late_fusion_added2_replay/`
  - `src/try/064_added_sota_added2_replay/`

当前跨域对照结果显示：

- `final` 域内最好 direct learned：
  - `TinyTCN_multiscale_late_fusion_2s_8s`
  - `case_mae = 0.1685`
- `added 21-24`：
  - `rpm_knn4`
  - `case_mae = 0.2293`
- `added2 25-30`：
  - `rpm_knn4`
  - `case_mae mean = 0.8131`
- `added2 25-30 | direct TinyTCN 2s+8s`
  - `case_mae = 2.2897`
- `added2 25-30 | rpm_knn4 + TinyTCN all_channels midband @ w=0.3`
  - `case_mae mean = 1.0126`

这说明：

- `TinyTCN 2s+8s` 仍是 `final` 域内最高精度模型；
- 但如果必须给出一个同时考虑 `final / added / added2` 的统一最终主干，当前更稳的默认选择仍是 `rpm_knn4`；
- `midband`、gate、prototype 和 alignment 当前都更适合作为研究型增强分支，而不是统一默认主线。

### 5.32 [2026-04-08] 若真实 `rpm` 不可直接使用，当前默认 deployable 风速链应优先参考 `fft_peak_1x_whole -> rpm_knn4`

- 状态：`current`
- 数据范围：
  - `final` 带标签工况 `1, 3-20`
  - `added` 带标签工况 `21-24`
- 代码口径：
  - `src/try/043_pred_rpm_deployability_check/`
  - `src/try/043_2_fft_rpm_to_wind_replay/`

当前可部署链对照结果显示：

- `added | fft_peak_1x_whole -> rpm_knn4`
  - `case_mae = 0.1860`
- `added | pred_rpm_2.0s -> rpm_linear`
  - `case_mae = 1.8886`
- `final | fft_window_peak_1x_conf_8s -> rpm_knn4`
  - `case_mae = 0.4148`
- `final | pred_rpm_2.0s -> rpm_linear`
  - `case_mae = 0.3917`

这说明：

- `pred_rpm -> wind` 在 `final` 旧域内仍可工作，但在外部域上已经不具备默认部署价值；
- FFT 解析 RPM 支线虽然在 `final` 旧域略弱于最佳 `pred_rpm` 链，但在 `added` 外部域明显更稳；
- 因此若真实 `rpm` 不可直接使用，当前更合理的默认 deployable 风速链应优先参考 `fft_peak_1x_whole -> rpm_knn4`。

### 5.33 [2026-04-08] sparse residual MoE V1 当前只在 `final` 域出现弱正信号，`added` 外部域仍不如 `rpm_knn4`

- 状态：`current`
- 数据范围：
  - `final`：带标签工况全量 `LOCO`
  - `added`：`工况21-24`
- 代码口径：
  - `src/try/065_sparse_router_residual_moe_v1/`
- 证据入口：
  - `outputs/kaggle_publish/065_sparse_router_residual_moe_v1/kernel_output_v3_retry/windywind/outputs/try/065_sparse_router_residual_moe_v1/summary_by_domain.csv`
  - `outputs/kaggle_publish/065_sparse_router_residual_moe_v1/kernel_output_v3_retry/windywind/outputs/try/065_sparse_router_residual_moe_v1/router_activation_table.csv`

当前 Kaggle GPU full run 结果显示：

- `final | A3_sparse_router_moe`
  - `case_mae = 0.4056`
- `final | rpm_knn4`
  - `case_mae = 0.4256`
- `added | A3_sparse_router_moe`
  - `case_mae = 0.4955`
- `added | rpm_knn4`
  - `case_mae = 0.2293`
- `A3_sparse_router_moe`
  - `final worse_than_base_rate = 0.3158`
- `A4_sparse_router_moe_without_noharm`
  - `final worse_than_base_rate = 0.6316`

这说明：

- unified sparse residual-MoE 在 `final` 域内已经有可保留的弱正信号；
- `L_noharm` 确实能够降低 `final` 侧伤害；
- 但当前 router 在 `added` 上仍过度偏向 `prototype expert`，导致外部域结果明显退化；
- 因此 `065` 当前更适合作为“继续加 trust / abstain 约束”的研究型候选，而不是统一默认主线。

## 6. 专题入口

- 数据目录与 manifest：`Docs/data_catalog.md`
- 表格主线：`Docs/table_pipeline.md`
- 风速回归实验总报告：`Docs/wind_speed_regression_experiment_report_2026-04-08.md`
- 数据质量与稳定结论：`Docs/data_quality_and_findings.md`
- added 异质分支梳理：`Docs/added_heterogeneity_branch_map_2026-04-07.md`
- 视频 RPM 主线：`Docs/video_pipeline.md`
- 视频 RPM CV 细节：`Docs/video_rpm_cv_pipeline.md`
- 视频手工标注资产：`Docs/video_manual_annotation_assets.md`
- 路线与方法备注：`Docs/project_roadmap_and_method_notes.md`
- 可部署融合路线验证计划：`Docs/deployable_fusion_validation_plan_2026-04-07.md`
- 视频设计备注：`Docs/video_module_design_notes.md`
- 第三阶段 CNN / TCN full 复核：`Docs/experiments/phase3_cnn_tcn_full_2026-04-05.md`
- TinyTCN 转速回归探索：`Docs/experiments/tinytcn_rpm_regression_2026-04-06.md`
- TinyTCN rpm 细窗长扫描：`Docs/experiments/tinytcn_rpm_fine_window_scan_2026-04-06.md`
- TinyTCN 边界窗口误差检查：`Docs/experiments/tinytcn_boundary_error_check_2026-04-06.md`
- TinyTCN 第一优先级快速验证：`Docs/experiments/tinytcn_priority1_quickcheck_2026-04-06.md`
- TinyTCN RPM 与风速窗长对照备注：`Docs/experiments/tinytcn_rpm_vs_wind_window_reference_2026-04-07.md`
- 双流 TinyTCN 快速验证：`Docs/experiments/dualstream_tinytcn_quickcheck_2026-04-07.md`
- 输入通道注意力 TinyTCN 快速验证：`Docs/experiments/input_channel_attention_tinytcn_quickcheck_2026-04-07.md`
- 后卷积通道注意力 TinyTCN 快速验证：`Docs/experiments/postconv_channel_attention_tinytcn_quickcheck_2026-04-07.md`
- 工况机制聚类探索：`Docs/experiments/case_mechanism_clustering_2026-04-07.md`
- 工况误差模式聚类探索：`Docs/experiments/case_error_mode_clustering_2026-04-07.md`
- 机制簇内 / 跨簇泛化快速验证：`Docs/experiments/cluster_generalization_quickcheck_2026-04-07.md`
- added 外部验证与可疑标签检查：`Docs/experiments/added_validation_label_check_2026-04-07.md`
- added 外部验证（包含难工况训练池）：`Docs/experiments/added_validation_with_full_final_pool_2026-04-07.md`
- added 反常表现域诊断：`Docs/experiments/added_domain_diagnosis_2026-04-07.md`
- 工况22 标签链路与模态外部对照：`Docs/experiments/case22_label_and_modality_check_2026-04-07.md`
- 应变侧漂移缓解快速验证：`Docs/experiments/strain_shift_mitigation_check_2026-04-07.md`
- 应变可迁移频带筛选：`Docs/experiments/strain_transfer_band_scan_2026-04-07.md`
- 中频应变细扫与融合权重验证：`Docs/experiments/midband_strain_weight_scan_2026-04-07.md`
- 解析基线与 Learned 中频分支复核：`Docs/experiments/rpm_vs_learned_midband_check_2026-04-07.md`
- `rpm_knn4 + learned midband` 多随机种子稳定性复核：`Docs/experiments/rpm_learned_midband_multiseed_stability_check_2026-04-07.md`
- `pred_rpm` 可部署性验证：`Docs/experiments/pred_rpm_deployability_check_2026-04-07.md`
- FFT 解析 RPM 算法搜索：`Docs/experiments/fft_rpm_algorithm_search_2026-04-07.md`
- FFT RPM -> Wind 结果回放：`Docs/experiments/fft_rpm_to_wind_replay_2026-04-07.md`
- FFT RPM 与 learned midband 融合回放：`Docs/experiments/fft_midband_fusion_replay_2026-04-07.md`
- added 并入训练池后的 2s/8s 统一 LOCO quickcheck：`Docs/experiments/added_in_training_loco_quickcheck_2026-04-07.md`
- true rpm 主干 + acc residual quickcheck：`Docs/experiments/true_rpm_acc_residual_quickcheck_2026-04-08.md`
- soft gate quickcheck：`Docs/experiments/soft_gate_quickcheck_2026-04-08.md`
- conservative gate quickcheck：`Docs/experiments/conservative_gate_quickcheck_2026-04-08.md`
- TCN soft gate persist：`Docs/experiments/tcn_soft_gate_persist_2026-04-08.md`
- TCN conservative gate persist：`Docs/experiments/tcn_conservative_gate_persist_2026-04-08.md`
- TinyTCN embedding kNN residual quickcheck：`Docs/experiments/tcn_embedding_knn_residual_quickcheck_2026-04-08.md`
- TinyTCN embedding 窗长信号 quickcheck：`Docs/experiments/tcn_embedding_window_signal_quickcheck_2026-04-08.md`
- support-window residual quickcheck：`Docs/experiments/support_window_residual_quickcheck_2026-04-08.md`
- case evidence aggregator quickcheck：`Docs/experiments/case_evidence_aggregator_quickcheck_2026-04-08.md`
- local mechanism router aligned head quickcheck：`Docs/experiments/local_mechanism_router_aligned_head_quickcheck_2026-04-08.md`
- embedding top-k local prototype fusion quickcheck：`Docs/experiments/embedding_topk_local_prototype_fusion_quickcheck_2026-04-08.md`
- embedding space diagnosis：`Docs/experiments/embedding_space_diagnosis_2026-04-08.md`
- prototype head ablation quickcheck：`Docs/experiments/prototype_head_ablation_quickcheck_2026-04-08.md`
- delta-only gate / bucket / trigger quickcheck：`Docs/experiments/delta_only_gate_bucket_trigger_quickcheck_2026-04-08.md`
- embedding top-k prototype alignment quickcheck：`Docs/experiments/embedding_topk_prototype_alignment_quickcheck_2026-04-08.md`
- prototype alignment SOTA recheck：`Docs/experiments/prototype_alignment_sota_recheck_2026-04-08.md`
- sparse residual MoE V1 Kaggle full run：`Docs/experiments/sparse_router_residual_moe_v1_kaggle_full_2026-04-08.md`
- final 训练的 `2s+8s` 晚融合外推 added2：`Docs/experiments/final_late_fusion_added2_replay_2026-04-08.md`
- added SOTA 回放 added2：`Docs/experiments/added_sota_added2_replay_2026-04-08.md`
- 模态参数识别探索线 v1：`Docs/experiments/modal_parameter_identification_v1_2026-04-08.md`

## 7. 关键工程约定

- 状态：`current`
- 最近复核：`2026-04-08`

当前约定：

- `PROJECT.md` 只保留当前仍有效的状态与结论
- 历史结论、被替代结论、教学性说明统一写入 `Docs/`
- 所有结论必须带绝对日期
- 避免使用“目前 / 现在 / 最近 / 这次”这类相对时间词
- 训练、复用或新增可持久化模型资产后，必须在 `PROJECT.md` 记录当前仍有效的模型权重、归一化参数、元数据或配置文件入口；仅作为历史对照的模型资产写入 `Docs/`
