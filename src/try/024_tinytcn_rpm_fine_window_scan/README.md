# 024 TinyTCN 转速回归细窗长扫描

## 目标

- 在一个新的 try 中，复用 `2026-04-06` 已实现的 TinyTCN 转速回归口径；
- 对 `2s` 到 `5s` 之间做更细粒度的窗长扫描；
- 回答两个问题：
  - 更小时间窗口是否会更好；
  - 还是 `2s-5s` 之间某个中间值更好。

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

- `2.0s`：`100` 点，步长 `50` 点
- `2.5s`：`125` 点，步长 `62` 点
- `3.0s`：`150` 点，步长 `75` 点
- `3.5s`：`175` 点，步长 `88` 点
- `4.0s`：`200` 点，步长 `100` 点
- `4.5s`：`225` 点，步长 `112` 点
- `5.0s`：`250` 点，步长 `125` 点

## 运行方式

```powershell
uv run python src/try/024_tinytcn_rpm_fine_window_scan/run_tinytcn_rpm_fine_window_scan.py --mode full
```

## 输出

- 输出目录：`outputs/try/024_tinytcn_rpm_fine_window_scan/`
- 固定产物：
  - `rpm_fine_window_scan_summary.csv`
  - `rpm_fine_window_scan_case_level_predictions.csv`
  - `rpm_fine_window_scan_unlabeled_predictions.csv`
  - `summary.md`
