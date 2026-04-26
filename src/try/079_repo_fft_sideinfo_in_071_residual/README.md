# 079 repo FFT side-info in 071 residual

## 目标

- 保持 `071` 的主干不变：
  - `rpm_knn4 + embedding residual ridge`
- 只把仓库 FFT side-info 追加到 residual ridge 输入里；
- 不改主干；
- 不改模型类型；
- 不加新 gate；
- 主排序口径只看：
  - `added_to_added2`
- 次级复核口径保留：
  - `added + added2 external LOOCV`

## 本次最小版输入特征

- `repo_fft_rpm`
- `repo_delta_rpm = repo_fft_rpm - true_rpm`
- `repo_abs_delta_rpm`
- `repo_fft_confidence`
- `repo_fft_source`
  - `fft_peak_1x_whole`
  - `window_peak_1x_conf_8s`

## 输入

- `outputs/try/069_added2_embedding_pca_projection/embedding_case_table.csv`
- `src/try/043_1_fft_rpm_algorithm_search/run_fft_rpm_algorithm_search.py`
- `src/try/078_repo_fft_rpm_blend_quickcheck/run_repo_fft_rpm_blend_quickcheck.py`

## 运行方式

先跑最小版：

```powershell
uv run python src/try/079_repo_fft_sideinfo_in_071_residual/run_repo_fft_sideinfo_in_071_residual.py
```

如果主口径出现正信号，再加 very small ablation：

```powershell
uv run python src/try/079_repo_fft_sideinfo_in_071_residual/run_repo_fft_sideinfo_in_071_residual.py --include-delta-only
```

## 输出

- 输出目录：`outputs/try/079_repo_fft_sideinfo_in_071_residual/`
- 主要文件：
  - `external_feature_table.csv`
  - `variant_feature_sets.csv`
  - `all_case_predictions.csv`
  - `summary_by_protocol.csv`
  - `summary_by_protocol_and_domain.csv`
  - `added_to_added2_compare_vs_071.csv`
  - `summary.md`
