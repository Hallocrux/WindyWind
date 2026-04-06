# 020 TinyTCN 转速回归窗长扫描

## 目标

- 在一个新的 try 中，复用 `2026-04-06` 已实现的 TinyTCN 转速回归口径；
- 保持监督目标为 `rpm`；
- 扫描不同窗长对工况级 `LOCO` 转速回归的影响。

## 输入与口径

- 数据入口：
  - `data/final/dataset_manifest.csv`
  - `data/final/datasets/工况*.csv`
- 清洗逻辑：复用 `src/current.data_loading`
- 原始窗口构造：复用 `src/try/012_phase3_end_to_end_shortlist/`
- TinyTCN rpm 训练与汇总：复用 `src/try/019_tinytcn_rpm_regression/`
- 评估方式：
  - 分组单位为工况
  - `Leave-One-Condition-Out`
- 监督目标：
  - `dataset_manifest.csv` 中的 `rpm`

## 默认扫描窗口

- `2s`：`100` 点，步长 `50` 点
- `4s`：`200` 点，步长 `100` 点
- `5s`：`250` 点，步长 `125` 点
- `8s`：`400` 点，步长 `200` 点

## 运行方式

开发态：

```powershell
uv run python src/try/020_tinytcn_rpm_window_length_scan/run_tinytcn_rpm_window_scan.py --mode dev
```

全量：

```powershell
uv run python src/try/020_tinytcn_rpm_window_length_scan/run_tinytcn_rpm_window_scan.py --mode full
```

## 输出

- 输出目录：`outputs/try/020_tinytcn_rpm_window_length_scan/`
- 固定产物：
  - `rpm_window_scan_summary.csv`
  - `rpm_window_scan_case_level_predictions.csv`
  - `rpm_window_scan_unlabeled_predictions.csv`
  - `summary.md`
