# 082 competition test inference

## 目标

- 对 `data/test/竞赛预测风速工况.csv` 进行一次独立推理探索；
- 复用 `2026-04-09` 已固定的 added-first 默认最佳模型 `071`；
- 同时输出若干当前已补测或已对齐过的候选参考：
  - `rpm_knn4`
  - `embedding_ridge`
  - `embedding_knn4`
  - `079 | rpm_knn4 + embedding + repo FFT side-info residual ridge`
  - `ridge_vib_ft_rpm`
  - `tabular_reference_g6_ridge`

## 输入

- `data/test/竞赛预测风速工况.csv`
- 测试工况给定 `rpm`
- `outputs/try/079_repo_fft_sideinfo_in_071_residual/external_feature_table.csv`
- `outputs/try/057_embedding_space_diagnosis/models/checkpoints/unified_all_2s.pt`
- `outputs/try/057_embedding_space_diagnosis/models/checkpoints/unified_all_8s.pt`
- `data/added/`
- `data/added2/`

## 默认口径

- `2026-04-09` 本探索默认把 `added + added2` 的带标签工况 `21-30` 作为最终训练池；
- `071` 与候选都在同一训练池上重拟合后，再对竞赛测试工况前向预测；
- `079` 仅作为 side-info 候选对照，不视为默认升级版。

## 运行方式

```powershell
uv run python src/try/082_competition_test_inference/run_competition_test_inference.py --rpm 204
```

## 输出

- 输出目录：`outputs/try/082_competition_test_inference/`
- 主要文件：
  - `prediction_summary.csv`
  - `prediction_summary.md`
  - `test_feature_row.csv`
  - `test_signal_inventory.json`

## 说明

- 若后续竞赛方提供更多已知条件，可继续在本探索目录上迭代；
- 本目录只做一次性测试推理，不修改正式主线。
