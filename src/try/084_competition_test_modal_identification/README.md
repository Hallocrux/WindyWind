# 084 competition test modal identification

## 目标

- 对 `data/test/竞赛预测频率工况.csv` 做一次独立的模态参数识别；
- 复用 `2026-04-09` 已正式沉淀到 `src/modal_parameter_identification/` 的清洗与识别口径；
- 不修改正式 manifest，也不把测试文件混入 `data/final/` 主数据。

## 输入

- `data/test/竞赛预测频率工况.csv`
- `data/final/dataset_manifest.csv`
- `data/final/datasets/工况*.csv`

## 默认口径

- 共有通道口径：复用 `final` 主线 `20` 个共有有效信号通道；
- 清洗口径：删除首尾连续缺失段；中间连续缺失 `<=5` 行线性插值，`>5` 行删除并切分连续段；
- 模态口径：`PSD / CSD / coherence / FDD / SSI-COV`；
- 默认采样率：`50 Hz`；
- 默认频带：预分析 `0.5-6.0 Hz`，关注 `2.0-3.0 Hz`。

## 运行方式

```powershell
uv run python src/try/084_competition_test_modal_identification/run_competition_test_modal_identification.py
```

如果已知工况级 `rpm`，可显式传入：

```powershell
uv run python src/try/084_competition_test_modal_identification/run_competition_test_modal_identification.py `
  --rpm 204
```

显式导出振型动画：

```powershell
uv run python src/try/084_competition_test_modal_identification/run_competition_test_modal_identification.py `
  --save-mode-shape-animation `
  --animation-format gif
```

## 输出

- 输出目录：`outputs/try/084_competition_test_modal_identification/`
- 主要文件：
  - `case_modal_summary.csv`
  - `window_modal_estimates.csv`
  - `harmonic_mask_table.csv`
  - `stabilization_poles.csv`
  - `stability_statistics.csv`
  - `strain_mode_shapes.csv`
  - `accy_mode_shapes.csv`
  - `modal_summary.md`
  - `signal_inventory.json`
  - `case_9001_modal_overview.png`
  - `case_9001_mode_shape_comparison.png`

## 说明

- `2026-04-09` 这份测试文件未随题面提供同步 `rpm` 时程，因此默认不做 `1x-4x` 谐波屏蔽；
- 若后续题面补充 `rpm` 或同步转速时程，可继续在本目录复跑。
