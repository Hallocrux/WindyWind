# 076 from_others multitask inference added series

## 目标

- 直接复用 `src/from_others/2/模型训练.py` 中的 `MultiTaskLSTM` 与现成权重 `wind_model.pth`；
- 不重训；
- 尽量按 `src/from_others/2/数据处理2.py` 的字段整理逻辑，把 `added`、`added2` 的标准 CSV 转成模型可直接推理的窗口；
- 输出 `added`、`added2` 的 case-level 风速与转速预测。

## 输入

- 外部来源模型定义：
  - `src/from_others/2/模型训练.py`
- 外部来源权重：
  - `src/from_others/2/wind_model.pth`
- 外部来源预处理规则参考：
  - `src/from_others/2/数据处理2.py`
- 数据：
  - `data/added/standardized_datasets/工况21-24.csv`
  - `data/added2/standardized_datasets/工况25-30.csv`
  - `data/added/dataset_manifest.csv`
  - `data/added2/dataset_manifest.csv`

## 方法摘要

- 目标输入列固定为 `20` 列：
  - `WSMS00001-00005` 的 `AccX/AccY/AccZ`
  - `应变传感器1-5.chdata`
- 复用原脚本规则：
  - 若存在 `WSMS00006.*`，则覆盖/补到 `WSMS00005.*`
  - 缺失列补 `0`
  - `NaN` 补 `0`
  - 滑窗：`100`
  - 步长：`50`
- 推理后按工况对全部窗口预测取平均，得到 case-level 风速和转速预测。

## 输出

- 输出目录：`outputs/try/076_from_others_multitask_inference_added_series/`
- 主要文件：
  - `case_level_predictions.csv`
  - `summary_by_domain.csv`
  - `summary.md`

## 运行方式

```powershell
uv run python src/try/076_from_others_multitask_inference_added_series/run_from_others_multitask_inference_added_series.py
```
