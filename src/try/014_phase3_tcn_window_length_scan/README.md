# 014 第三阶段 TinyTCN 窗长扫描

## 目标

- 在一个新的 try 中，只针对 `TinyTCN` 扫描不同窗长。
- 默认只用小数据集，快速判断哪个窗长更适合当前 TinyTCN 路线。
- 不再混入其他模型，避免实验矩阵继续膨胀。

## 固定实验列表

- `TinyTCN @ 2s`，步长 `1s`
- `TinyTCN @ 4s`，步长 `2s`
- `TinyTCN @ 5s`，步长 `2.5s`
- `TinyTCN @ 8s`，步长 `4s`

默认保持 `50Hz` 采样率和 `50%` overlap。

## 输入与口径

- 数据入口：
  - `data/final/dataset_manifest.csv`
  - `data/final/datasets/工况*.csv`
- 清洗逻辑：复用 `src/current.data_loading`
- 原始窗口构造：复用 `src/try/012_phase3_end_to_end_shortlist/`
- 模型逻辑：复用 `src/try/013_phase3_cnn_tcn_smoke/`
- 评估方式：
  - 分组单位为工况
  - `Leave-One-Condition-Out`

## 运行方式

开发态：

```powershell
uv run python src/try/014_phase3_tcn_window_length_scan/run_tcn_window_scan.py --mode dev
```

全量：

```powershell
uv run python src/try/014_phase3_tcn_window_length_scan/run_tcn_window_scan.py --mode full
```

显式指定工况：

```powershell
uv run python src/try/014_phase3_tcn_window_length_scan/run_tcn_window_scan.py --case-ids 1 2 3 5 15 16
```

## 输出

- 输出目录：`outputs/try/014_phase3_tcn_window_length_scan/`
- 固定产物：
  - `tcn_window_scan_summary.csv`
  - `tcn_window_scan_case_level_predictions.csv`
  - `summary.md`
