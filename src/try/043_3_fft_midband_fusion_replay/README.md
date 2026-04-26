# 043_3 FFT RPM 与 learned midband 融合回放

## 目标

- 不重训任何 `TinyTCN`，也不重跑任何 FFT RPM 识别；
- 直接复用：
  - `042` 已落盘的 `TinyTCN all_channels midband` 多 seed case 级预测；
  - `043_2` 已落盘的 `fft_rpm -> wind` case 级预测；
- 验证：在 `data/added/` 的 `工况21-24` 上，FFT 支线能否替代 `true_rpm` 解析支线，继续作为最终融合算法的一部分。

## 输入

- `outputs/try/042_rpm_learned_midband_multiseed_stability_check/seed_case_level_predictions.csv`
- `outputs/try/043_2_fft_rpm_to_wind_replay/fft_rpm_to_wind_case_level_predictions.csv`

## 运行方式

```powershell
uv run python src/try/043_3_fft_midband_fusion_replay/run_fft_midband_fusion_replay.py
```

## 输出

- 输出目录：`outputs/try/043_3_fft_midband_fusion_replay/`
- 主要文件：
  - `case_level_predictions.csv`
  - `seed_summary.csv`
  - `stability_overview.csv`
  - `pairwise_comparison.csv`
  - `best_variant_by_seed.csv`
  - `summary.md`

## 说明

- 本 try 只做结果回放式融合：
  - 读取以前 try 已经算好的 case 级预测；
  - 在当前脚本中完成加权融合、统计和对照；
  - 不重新计算上游识别结果。
