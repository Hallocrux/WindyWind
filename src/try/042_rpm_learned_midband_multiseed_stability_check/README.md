# 042 `rpm_knn4 + TinyTCN midband @ w=0.5` 多随机种子稳定性复核

## 目标

- 复核 `041` 中单次最优的 `rpm_knn4 + TinyTCN all_channels midband @ learned_weight=0.5`；
- 判断它是否只是某个 seed 的偶然最优，还是在多随机种子下依然稳定优于 `rpm_knn4`；
- 顺带检查 `w=0.5` 相比相邻权重 `0.3 / 0.7` 是否仍处在最优附近。

## 输入与口径

- 主数据：
  - `data/final/dataset_manifest.csv`
  - `data/final/datasets/工况*.csv`
- added 数据：
  - `data/added/dataset_manifest.csv`
  - `data/added/standardized_datasets/工况21-24.csv`
- 清洗逻辑：复用 `src/current.data_loading`
- learned 分支：
  - `TinyTCN@5s`
  - `all_channels + strain bandpass 3.0-6.0Hz`
- 训练池：
  - `full_final_pool`

## 运行方式

```powershell
.venv\Scripts\python.exe src/try/042_rpm_learned_midband_multiseed_stability_check/run_rpm_learned_midband_multiseed_stability_check.py
```

如需自定义种子：

```powershell
.venv\Scripts\python.exe src/try/042_rpm_learned_midband_multiseed_stability_check/run_rpm_learned_midband_multiseed_stability_check.py --seeds 42 52 62 72 82 92 102 112 122 132
```

## 输出

- 输出目录：`outputs/try/042_rpm_learned_midband_multiseed_stability_check/`
- 固定产物：
  - `variant_config_table.csv`
  - `seed_summary.csv`
  - `seed_case_level_predictions.csv`
  - `best_variant_by_seed.csv`
  - `stability_overview.csv`
  - `pairwise_comparison.csv`
  - `case22_by_seed.csv`
  - `summary.md`
