# 007 工况5 更稳中段 30s 逐列图像

## 目标

- 从 `工况5` 删除首尾连续缺失段后的稳定保留部分中，再截取中心 `30s` 作为“更稳的中段”。
- 把这 `30s` 的所有有效信号列逐列画出来，便于观察更稳阶段的波形形态。

## 输入与口径

- 数据文件：`data/final/datasets/工况5.csv`
- 清洗逻辑：复用 `src/current/data_loading.py`
- 当前截取规则：
  - 先删除首尾连续缺失段；
  - 在裁剪后的保留段上，以时间中心为中心截取 `30s`；
  - 对这 `30s` 的全部 `20` 个有效信号列做可视化。
- 当前对应时间范围：
  - `2026-03-30 16:25:09.010`
  - `2026-03-30 16:25:39.010`

## 输出

- 输出目录：`outputs/try/007_case5_core_middle_30s_plots/`
- 主要产物：
  - `case5_core_middle_30s_small_multiples.png`
  - `case5_core_middle_30s_acceleration_overlay.png`
  - `case5_core_middle_30s_strain_overlay.png`
  - `channel_summary.csv`
  - `summary.md`

## 运行方式

```powershell
uv run python src/try/007_case5_core_middle_30s_plots/plot_case5_core_middle_30s.py
```
