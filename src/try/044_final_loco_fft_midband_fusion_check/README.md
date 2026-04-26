# 044 final LOCO FFT + learned midband 融合补齐

## 目标

- 只补齐 `final LOCO` 下缺失的 `TinyTCN all_channels midband` case 级预测；
- 复用 `043_2` 已落盘的 FFT `final LOCO` `rpm -> wind` 结果；
- 复用 `final` manifest 的真实 `rpm`，补一条 `true_rpm + learned midband` 上界参考；
- 判断 FFT 支线在 `final LOCO` 上能否继续替代 `true_rpm`，进入融合候选。

## 输入

- `outputs/try/043_2_fft_rpm_to_wind_replay/fft_rpm_to_wind_case_level_predictions.csv`
- `data/final/dataset_manifest.csv`
- `data/final/datasets/工况{ID}.csv`

## 运行方式

```powershell
uv run python src/try/044_final_loco_fft_midband_fusion_check/run_final_loco_fft_midband_fusion_check.py
```

## 输出

- 输出目录：`outputs/try/044_final_loco_fft_midband_fusion_check/`
- 主要文件：
  - `case_level_predictions.csv`
  - `seed_summary.csv`
  - `stability_overview.csv`
  - `pairwise_comparison.csv`
  - `best_variant_by_seed.csv`
  - `summary.md`

## 说明

- 本 try 不重算 added 部分；
- FFT `final LOCO` 结果全部直接复用 `043_2`；
- 唯一新计算的是 `final LOCO` 的 learned midband 分支。
