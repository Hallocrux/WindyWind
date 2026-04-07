# 工况22 标签链路与模态外部对照（2026-04-07）

- 状态：`current`
- 首次确认：`2026-04-07`
- 最近复核：`2026-04-07`
- 数据范围：
  - 主训练域：`data/final/` 的带标签工况
  - added 外部工况：`data/added/` 的 `工况21-24`
  - 重点工况：`工况22`
- 代码口径：
  - `src/try/037_case22_label_and_modality_check/`
- 证据入口：
  - `outputs/try/037_case22_label_and_modality_check/file_copy_audit.csv`
  - `outputs/try/037_case22_label_and_modality_check/label_chain_audit.csv`
  - `outputs/try/037_case22_label_and_modality_check/modality_case_predictions.csv`
  - `outputs/try/037_case22_label_and_modality_check/modality_summary.csv`
  - `outputs/try/037_case22_label_and_modality_check/case22_modality_focus.csv`
  - `outputs/try/037_case22_label_and_modality_check/summary.md`

## 1. 目标

在 `036` 已经确认 `工况22` 是 added 中最异常的机制点后，进一步回答：

- `工况22` 的标准化副本是否忠实对应原始补充文件；
- 当前 added 外部高估主要由哪类输入驱动；
- `工况22` 标签是否已经有足够强的仓库内证据链。

## 2. 方法口径

- 文件审计：
  - 对 `data/added/datasets/*.csv`
  - 与 `data/added/standardized_datasets/工况21-24.csv`
  - 做大小与 `sha256` 一致性比对
- 标签链路审计：
  - 读取 `data/added/dataset_manifest.csv`
  - 用 `final` 的 RPM 邻居与线性参考做弱一致性检查
- 模态外部验证：
  - 模型：`TinyTCN@5s`
  - 训练池：
    - `full_final_pool`
    - `clean_final_pool`（去掉 `工况1 / 3 / 17 / 18`）
  - 输入模态：
    - `all_channels`
    - `strain_only`
    - `acc_only`
    - `rpm_linear / rpm_knn4`

## 3. 当前结果

### 3.1 [2026-04-07] added 原始文件与标准化副本是一致的字节级拷贝

- 状态：`current`
- 首次确认：`2026-04-07`
- 最近复核：`2026-04-07`

`21-24` 的文件审计结果显示：

- `raw_size_bytes == standard_size_bytes`
- `raw_sha256 == standard_sha256`
- `is_byte_identical_copy = 1`

这说明：

- `工况22` 的异常不是由“标准化复制过程损坏文件”导致的；
- 当前应继续把问题聚焦在标签链路和模态机制，而不是文件搬运错误。

### 3.2 [2026-04-07] `工况22` 的标签在 RPM 参考下并不离谱，但仓库内证据链仍不完整

- 状态：`current`
- 首次确认：`2026-04-07`
- 最近复核：`2026-04-07`

`工况22` 的 manifest 记录为：

- 原始文件名：`24-3补充工况2.csv`
- 标签来源：`人工核验（数据列表截图 2026-04-06）`
- 标签：`3.4 m/s`
- 转速：`106 rpm`

RPM 参考结果：

- `rpm_linear_pred_from_final = 2.3728`
- `rpm_knn4_pred_from_final = 3.1342`

同时当前仓库内：

- `has_local_label_source_artifact = 0`

这说明：

- `3.4 m/s @ 106 rpm` 在 RPM 近邻上不是完全不合理的孤立点；
- 但支撑该标签的“截图工件”并未落入仓库；
- 因此当前更合理的表述应是：
  - `工况22` 标签“人工核验但仓库内证据不足”
  - 而不是“已经被仓库内证据完全坐实”

### 3.3 [2026-04-07] added 外部高估主要由应变侧输入驱动

- 状态：`current`
- 首次确认：`2026-04-07`
- 最近复核：`2026-04-07`

模态汇总结果：

- `rpm_knn4`
  - `case_mae = 0.2293`
- `full_final_pool|acc_only`
  - `case_mae = 0.3553`
- `rpm_linear`
  - `case_mae = 0.4093`
- `clean_final_pool|acc_only`
  - `case_mae = 0.4362`
- `clean_final_pool|all_channels`
  - `case_mae = 2.0341`
- `clean_final_pool|strain_only`
  - `case_mae = 2.1299`
- `full_final_pool|strain_only`
  - `case_mae = 2.3854`
- `full_final_pool|all_channels`
  - `case_mae = 3.2158`

这说明：

- `acc_only` 在 `full / clean` 两个训练池下都远优于 `all_channels` 与 `strain_only`；
- 当前 added 外部高估的主来源不是加速度侧，而是应变侧输入；
- 把应变通道重新并入模型，会显著放大 added 的系统性高估。

### 3.4 [2026-04-07] `工况22` 在 `acc_only` 与 `rpm_knn4` 下都明显更合理

- 状态：`current`
- 首次确认：`2026-04-07`
- 最近复核：`2026-04-07`

`工况22` 的 case 级结果：

- `rpm_knn4`
  - `pred = 3.1342`
  - `abs_error = 0.2658`
- `full_final_pool|acc_only`
  - `pred = 3.7484`
  - `abs_error = 0.3484`
- `clean_final_pool|acc_only`
  - `pred = 4.6221`
  - `abs_error = 1.2221`
- `clean_final_pool|all_channels`
  - `pred = 6.7770`
  - `abs_error = 3.3770`
- `clean_final_pool|strain_only`
  - `pred = 6.8429`
  - `abs_error = 3.4429`
- `full_final_pool|strain_only`
  - `pred = 7.0698`
  - `abs_error = 3.6698`
- `full_final_pool|all_channels`
  - `pred = 8.1923`
  - `abs_error = 4.7923`

这说明：

- `工况22` 的极端高估并不是 added 全体不可避免的结果；
- 只要不让应变侧主导，预测就会显著回到更合理区间；
- 因此 `工况22` 当前更像：
  - “应变机制失配”
  - 而不是“标签一定错了”

## 4. 当前判断

`2026-04-07` 的这轮检查支持以下判断：

- `工况22` 的文件复制链路没有问题；
- `工况22` 标签与 RPM 近邻并不强冲突，但仓库内缺少 `label_source` 对应的截图工件；
- added 外部失配的主驱动已经更明确地指向应变侧域偏移；
- 如果后续继续推进，优先级应是：
  - 先补 `工况22` 的本地标签证据工件或来源说明；
  - 再做“应变侧归一化 / 频带屏蔽 / 域适配”类对照；
  - 而不是继续无条件扩大训练池。
