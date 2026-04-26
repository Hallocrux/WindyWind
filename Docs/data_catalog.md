# 数据目录与工况资产

本文档记录当前数据资产的组织方式、字段口径和人工维护信息。  
如无特殊说明，本文档中的“当前”均指最近一次明确复核日期。

## 1. 当前有效数据组织

- 状态：`current`
- 首次确认：`2026-04-05`
- 最近复核：`2026-04-05`

当前表格主线使用下面两部分共同组织数据：

- `data/final/datasets/工况{ID}.csv`
  - 宽表数据文件
  - 文件名只保留工况编号
- `data/final/dataset_manifest.csv`
  - 宽表数据的唯一人工元数据来源
  - 程序不再从文件名解析 `case_id / wind_speed / rpm`

## 2. 数据目录说明

### 2.1 `data/test_data_with_explanation/`

- 状态：`current`
- 首次确认：`2026-03-30`
- 最近复核：`2026-04-05`

用途：

- 保存较原始的长表数据
- 每行是一条单传感器采样记录

当前字段：

- `code`
- `type`
- `time`
- `value1`
- `value2`
- `value3`

当前映射：

- `type=vibr`
  - `value1 -> AccX`
  - `value2 -> AccY`
  - `value3 -> AccZ`
- `type=sgd`
  - `value1 -> chdata`

### 2.2 `data/final/datasets/`

- 状态：`current`
- 首次确认：`2026-03-30`
- 最近复核：`2026-04-05`

用途：

- 保存整理后的宽表数据
- 每行代表同一时刻的多传感器拼接结果

当前固定规则：

- 文件名统一为 `工况{ID}.csv`
- 物理文件名不再承载风速、转速、组别、补充工况别名等业务信息

### 2.3 `data/final/dataset_manifest.csv`

- 状态：`current`
- 首次确认：`2026-04-05`
- 最近复核：`2026-04-05`
- 代码口径：`src/current.data_loading`

用途：

- 作为宽表数据的唯一人工元数据源
- 保存不能从 CSV 自动推导、且需要人工核验的字段

当前字段：

- `case_id`
- `display_name`
- `wind_speed`
- `rpm`
- `original_file_name`
- `label_source`
- `notes`

当前规则：

- `case_id` 唯一
- 标准文件名由 `case_id` 派生为 `工况{ID}.csv`
- `wind_speed`、`rpm` 允许为空
- `is_labeled` 由程序派生，不在 manifest 手填
- `file_name`、`file_path` 由程序派生，不在 manifest 手填

### 2.4 `data/added/`

- 状态：`current`
- 首次确认：`2026-04-07`
- 最近复核：`2026-04-07`

用途：

- 暂存 `2026-04-06` 新补充的 `4` 个宽表工况
- 在正式复核与主线合并前，与 `data/final/` 分离管理

当前组成：

- `data/added/datasets/`
  - 保留补充工况的原始文件名
- `data/added/standardized_datasets/`
  - 保存按 `工况{ID}.csv` 复制的标准化副本
- `data/added/dataset_manifest.csv`
  - 保存 `case_id=21-24` 的人工元数据

当前映射：

- `工况21.csv`
  - 对应 `24-3补充工况.csv`
- `工况22.csv`
  - 对应 `24-3补充工况2.csv`
- `工况23.csv`
  - 对应 `24-4补充工况.csv`
- `工况24.csv`
  - 对应 `24-4补充工况2.csv`

### 2.5 `data/added2/`

- 状态：`current`
- 首次确认：`2026-04-08`
- 最近复核：`2026-04-08`

用途：

- 暂存 `2026-04-08` 新补充的 `6` 个宽表工况
- 在正式复核与主线合并前，与 `data/final/`、`data/added/` 分离管理

当前组成：

- `data/added2/`
  - 保留用户给出的原始文件名
- `data/added2/standardized_datasets/`
  - 保存按 `工况{ID}.csv` 复制的标准化副本
- `data/added2/dataset_manifest.csv`
  - 保存 `case_id=25-30` 的人工元数据

当前映射：

- `工况25.csv`
  - 对应 `23-2补充工况.csv`
- `工况26.csv`
  - 对应 `23-2补充工况2.csv`
- `工况27.csv`
  - 对应 `23-2补充工况3.csv`
- `工况28.csv`
  - 对应 `23-2补充工况4.csv`
- `工况29.csv`
  - 对应 `23-2补充工况5.csv`
- `工况30.csv`
  - 对应 `23-2补充工况6.csv`

## 3. 当前工况资产

- 状态：`current`
- 首次确认：`2026-04-05`
- 最近复核：`2026-04-05`
- 数据范围：`工况1` 到 `工况20`
- 证据入口：
  - `data/final/dataset_manifest.csv`
  - `outputs/dataset_inventory.csv`

当前共有 `20` 个工况：

- 其中 `19` 个带有 `wind_speed` 与 `rpm`
- `工况2.csv` 当前无标签

新增工况编号口径：

- `工况15.csv`、`工况16.csv`、`工况17.csv` 为新增工况
- 原 `24-2补充工况.csv`、`24-2补充工况2.csv`、`24-2补充工况3.csv`
  - 在当前主线中分别对应 `工况18.csv`、`工况19.csv`、`工况20.csv`
- 旧文件名仅保留在 manifest 的 `original_file_name` 中用于追溯

## 4. 宽表字段与异常列

- 状态：`current`
- 首次确认：`2026-04-02`
- 最近复核：`2026-04-05`
- 数据范围：`工况1` 到 `工况20`
- 代码口径：`src/current.data_loading`

当前宽表固定主键列：

- `time`

当前信号列通常包括：

- `WSMS00001.AccX/Y/Z`
- `WSMS00002.AccX/Y/Z`
- `WSMS00003.AccX/Y/Z`
- `WSMS00004.AccX/Y/Z`
- `WSMS00006.AccX/Y/Z`
- `应变传感器1.chdata` 到 `应变传感器5.chdata`

当前已确认的列规模：

- 大多数工况有 `21` 列
- `工况5.csv`、`工况12.csv`、`工况13.csv`、`工况14.csv` 有 `24` 列

### 4.1 已知错误列

- 状态：`current`
- 首次确认：`2026-04-02`
- 最近复核：`2026-04-05`
- 数据范围：`工况5`、`工况12`、`工况13`、`工况14`
- 证据入口：
  - `outputs/dataset_inventory.csv`
  - `outputs/data_quality_summary.csv`

当前确认：

- 所有 `WSMS00005.AccX/Y/Z` 都视为错误数据
- 正确的第 5 个加速度传感器实际在 `WSMS00006.AccX/Y/Z`
- 当前正式清洗必须显式忽略 `WSMS00005.*`

## 5. 自动派生资产

- 状态：`current`
- 首次确认：`2026-04-05`
- 最近复核：`2026-04-05`

下列信息不在 manifest 手工维护，而是由程序自动扫描得到：

- 行数 `row_count`
- 列数 `column_count`
- 开始时间 `start_time`
- 结束时间 `end_time`
- 时长 `duration_seconds`
- 估计采样率 `sampling_hz_est`
- 是否含 `WSMS00005` 错误列

当前默认输出位置：

- `outputs/dataset_inventory.csv`

## 6. 相关入口

- 表格主线说明：`Docs/table_pipeline.md`
- 数据质量与稳定结论：`Docs/data_quality_and_findings.md`
- 当前项目状态总览：`PROJECT.md`
