# 表格主线说明

本文档描述当前表格主线的数据处理、特征构造、模型评估和无标签推理规则。

## 1. 当前默认入口

- 状态：`current`
- 首次确认：`2026-04-02`
- 最近复核：`2026-04-05`

当前默认入口：

- `main.py`
  - 调用 `src/current.pipeline.main`

当前主线目录：

- `src/current/`

## 2. 当前数据入口与清洗口径

- 状态：`current`
- 首次确认：`2026-04-05`
- 最近复核：`2026-04-05`
- 数据范围：`工况1` 到 `工况20`
- 代码口径：
  - `src/current.data_loading`
  - `src/current.features`

当前数据入口：

- manifest：`data/final/dataset_manifest.csv`
- 宽表文件：`data/final/datasets/工况{ID}.csv`

当前清洗流程：

1. 从 manifest 读取 `case_id / wind_speed / rpm / display_name`
2. 按 `case_id` 推导标准文件名
3. 读取 CSV，清理 `time` 字段文本格式
4. 解析时间、排序、按时间去重
5. 只保留所有工况共有的有效信号列
6. 显式忽略 `WSMS00005.*`
7. 删除首尾连续缺失段
8. 对中间连续缺失 `<=5` 行的缺失块做线性插值，再做 `ffill / bfill`
9. 对中间连续缺失 `>5` 行的缺失块直接删除
10. 滑动窗口只在连续段内部生成

## 3. 窗口与特征

- 状态：`current`
- 首次确认：`2026-04-02`
- 最近复核：`2026-04-05`
- 代码口径：`src/current.features`

当前窗口参数：

- 采样率：`50 Hz`
- 窗长：`250` 点，即 `5s`
- 步长：`125` 点，即 `2.5s`

当前切窗规则补充：

- 每个 `__segment_id` 连续段独立滑窗
- 窗口不会跨越被长缺失切开的段边界
- `raw_missing_ratio / raw_missing_rows` 只统计保留下来的短缺失插值行

当前每个窗口输出：

- 元数据：
  - `case_id`
  - `file_name`
  - `window_index`
  - `start_time`
  - `end_time`
  - `wind_speed`
  - `rpm`
- 质量字段：
  - `raw_missing_ratio`
  - `raw_missing_rows`
  - `touches_leading_missing`
  - `touches_trailing_missing`
- 通道特征：
  - `mean`
  - `std`
  - `min`
  - `max`
  - `ptp`
  - `rms`
  - `fft_peak_freq`
  - `fft_peak_amp`
  - `fft_total_energy`
  - `0-2Hz / 2-5Hz / 5-10Hz` 频带能量占比

## 4. Train / Eval / Inference 规则

- 状态：`current`
- 首次确认：`2026-04-02`
- 最近复核：`2026-04-05`
- 数据范围：带标签工况 `19` 个，无标签工况 `1` 个
- 代码口径：`src/current.experiment`

当前实验不是固定的 `train / eval / test` 三块切分，而是：

- 在全部带标签工况上做 `Leave-One-Condition-Out`
- 对无标签工况单独做最终推理

### 4.1 Train / Eval

每一轮交叉验证的分组单位是“工况”，不是“窗口”：

- `train`
  - 当前被留出工况之外的其余带标签工况
- `eval`
  - 当前被留出的那个带标签工况

当前原因：

- 同一工况切出的窗口高度相似
- 若把窗口随机打散到 train 和 eval，会发生明显信息泄漏

### 4.2 Test

当前没有单独、带真实标签的独立 test set。

因此当前用于模型比较的评估结果，全部来自：

- 带标签工况上的留一交叉验证

### 4.3 无标签推理

`工况2.csv` 当前无标签，因此：

- 不参与 train
- 不参与 eval 打分
- 只在模型选好后做最终 inference

当前兼容规则：

- 如果全局最优模型依赖 `rpm`
- 且无标签工况缺少 `rpm`
- 则自动回退到最佳 `rpm-free` 模型

## 5. 当前模型集合与输出

- 状态：`current`
- 首次确认：`2026-04-02`
- 最近复核：`2026-04-05`

当前比较的模型集合：

- `LinearRegression + RPM_ONLY`
- `Ridge + VIB_FT`
- `RandomForestRegressor + VIB_FT`
- `HistGradientBoostingRegressor + VIB_FT`
- `Ridge + VIB_FT_RPM`
- `RandomForestRegressor + VIB_FT_RPM`
- `HistGradientBoostingRegressor + VIB_FT_RPM`

当前默认输出：

- `outputs/model_summary.csv`
- `outputs/case_level_predictions.csv`
- `outputs/window_level_predictions.csv`
- `outputs/unlabeled_predictions.csv`
- `outputs/data_quality_summary.csv`
- `outputs/data_quality_missing_columns.csv`
- `outputs/dataset_inventory.csv`

## 6. 当前稳定结论

- 状态：`current`
- 首次确认：`2026-04-05`
- 最近复核：`2026-04-05`
- 数据范围：manifest 迁移后的 `工况1` 到 `工况20`
- 代码口径：`src/current/`
- 证据入口：
  - `outputs/model_summary.csv`
  - `outputs/unlabeled_predictions.csv`

当前主流程共生成：

- `866` 个窗口样本
- 带标签工况 `19` 个
- 无标签工况 `1` 个

当前最优模型：

- 若允许使用 `rpm`：`Ridge + VIB_FT_RPM`
- 若必须 rpm-free：`Ridge + VIB_FT`

当前 `工况2.csv` 的无标签预测结果：

- `3.3279 m/s`

## 7. 相关入口

- 数据目录与 manifest：`Docs/data_catalog.md`
- 数据质量与稳定结论：`Docs/data_quality_and_findings.md`
- 当前项目状态总览：`PROJECT.md`
