# 043_1 FFT 解析 RPM 算法搜索

## 目标

- 按解析方法重建一条不依赖监督训练的 `sensor -> pred_rpm` 支线；
- 系统比较以下 FFT / 周期估计候选：
  - 单峰 `1x` / `3x` 基线；
  - 多谐波模板打分；
  - 峰值回投票；
  - 频谱打分 + 自相关先验消歧；
  - 整段估计与滑窗聚合估计。
- 重点回答：
  - added 外部域上，哪种 FFT 方向的 RPM 算法最稳；
  - 当前误差主要来自“找不到转动频带”还是“倍频阶次混淆”；
  - 解析 RPM 是否有机会替代 `043` 中失效的 learned `pred_rpm`。

## 输入与口径

- 主数据：
  - `data/final/dataset_manifest.csv`
  - `data/final/datasets/工况*.csv`
- added 数据：
  - `data/added/dataset_manifest.csv`
  - `data/added/standardized_datasets/工况21-24.csv`
- 清洗逻辑：复用 `src/current.data_loading`
- 默认只使用加速度通道：
  - `WSMS*.AccX`
  - `WSMS*.AccY`
  - `WSMS*.AccZ`
- 默认 RPM 搜索频带：
  - `1.2Hz - 4.8Hz`
  - 即约 `72 - 288 rpm`

## 默认候选算法

- `fft_peak_1x_whole`
- `fft_peak_1x_welch_8s`
- `fft_peak_1x_welch_12s`
- `fft_peak_3x_whole`
- `window_peak_1x_conf_8s`
- `window_peak_1x_conf_12s`
- `hybrid_peak_1x_whole_window8_gate150`
- `fft_vote_whole`
- `harmonic_template_whole`
- `harmonic_template_autocorr_whole`
- `harmonic_template_welch_12s`
- `harmonic_template_autocorr_welch_12s`
- `window_template_conf_12s`
- `window_vote_conf_12s`
- `window_template_autocorr_conf_12s`

## 运行方式

```powershell
uv run python src/try/043_1_fft_rpm_algorithm_search/run_fft_rpm_algorithm_search.py
```

如需只跑部分变体：

```powershell
uv run python src/try/043_1_fft_rpm_algorithm_search/run_fft_rpm_algorithm_search.py --variants harmonic_template_autocorr_welch_12s window_template_autocorr_conf_12s
```

## 输出

- 输出目录：`outputs/try/043_1_fft_rpm_algorithm_search/`
- 固定产物：
  - `variant_config_table.csv`
  - `case_level_predictions.csv`
  - `summary.csv`
  - `best_variant_by_domain.csv`
  - `failure_cases.csv`
  - `summary.md`
