# 019 TinyTCN 转速回归

## 目标

- 在一个新的 try 中，复用 `2026-04-06` 现有第三阶段原始时序口径；
- 继续使用 `TinyTCN`；
- 把监督目标从 `wind_speed` 切换为 `rpm`，做工况级转速回归。

## 输入与口径

- 数据入口：
  - `data/final/dataset_manifest.csv`
  - `data/final/datasets/工况*.csv`
- 清洗逻辑：复用 `src/current.data_loading`
- 原始窗口构造：复用 `src/try/012_phase3_end_to_end_shortlist/`
- 默认窗口参数：
  - 采样率 `50Hz`
  - 窗长 `5s`
  - 步长 `2.5s`
- 评估方式：
  - 分组单位为工况
  - `Leave-One-Condition-Out`
- 监督目标：
  - `dataset_manifest.csv` 中的 `rpm`

## 运行方式

开发态：

```powershell
uv run python src/try/019_tinytcn_rpm_regression/run_tinytcn_rpm_regression.py --mode dev
```

全量：

```powershell
uv run python src/try/019_tinytcn_rpm_regression/run_tinytcn_rpm_regression.py --mode full
```

显式指定工况：

```powershell
uv run python src/try/019_tinytcn_rpm_regression/run_tinytcn_rpm_regression.py --case-ids 1 2 3 5 15 16
```

## 输出

- 输出目录：`outputs/try/019_tinytcn_rpm_regression/`
- 固定产物：
  - `model_summary.csv`
  - `case_level_predictions.csv`
  - `unlabeled_predictions.csv`
  - `summary.md`

## 当前默认训练配置

- 设备：CPU
- batch size：`32`
- 最大 epoch：`40`
- early stopping patience：`8`
- 学习率：`1e-3`
