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

### 5.13 [2026-04-07] added 方向当前更稳的默认候选为 `rpm_knn4 + learned midband @ learned_weight≈0.3`

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

- 当前 added 方向最稳的路线已经确定为“解析基线 + learned 中频分支”的混合方案；
- `rpm_knn4` 与 `learned midband` 的关系更像互补，而不是替代；
- 若目标是 `工况21-24` 的整体平均误差，当前默认固定权重应优先参考更稳的 `0.3`；
- 若目标是尽量修平 `工况22` 这类 hardest case，`0.5` 仍可作为参考权重保留。

## 6. 专题入口

- 数据目录与 manifest：`Docs/data_catalog.md`
- 表格主线：`Docs/table_pipeline.md`
- 数据质量与稳定结论：`Docs/data_quality_and_findings.md`
- 视频 RPM 主线：`Docs/video_pipeline.md`
- 视频 RPM CV 细节：`Docs/video_rpm_cv_pipeline.md`
- 视频手工标注资产：`Docs/video_manual_annotation_assets.md`
- 路线与方法备注：`Docs/project_roadmap_and_method_notes.md`
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

## 7. 关键工程约定

- 状态：`current`
- 最近复核：`2026-04-05`

当前约定：

- `PROJECT.md` 只保留当前仍有效的状态与结论
- 历史结论、被替代结论、教学性说明统一写入 `Docs/`
- 所有结论必须带绝对日期
- 避免使用“目前 / 现在 / 最近 / 这次”这类相对时间词
