# 089 non-causal CNN encoder ablation

## 目标

基于 `071` 的 added-first 下游口径，做一个更窄的 encoder 消融：

- 保留 residual block；
- 保留每个 block 两层 `Conv1d`；
- 保留全局平均池化；
- 保持每个窗口 encoder 的 embedding 维度为 `32`；
- 保持 `2s + 8s` 拼接后的 case embedding 维度为 `64`；
- 保持下游 `rpm_knn4 + embedding residual ridge` 管线兼容；
- 只把 causal crop 去掉，改为 non-causal 对称 padding；
- 提供 `with_dilation=True/False` 两个 encoder 变体。

核心问题：

- `071` 的收益是否来自 causal convolution；
- 还是主要来自 dilation、residual block、两层卷积结构和更大感受野。

## 运行方式

```powershell
uv run python src/try/089_noncausal_cnn_encoder_ablation/run_noncausal_cnn_encoder_ablation.py
```

可选快速复跑：

```powershell
uv run python src/try/089_noncausal_cnn_encoder_ablation/run_noncausal_cnn_encoder_ablation.py --max-epochs 12 --force-retrain
```

导出 `noncausal_dilated` 全部工况 PCA 二维图：

```powershell
uv run python src/try/089_noncausal_cnn_encoder_ablation/project_noncausal_dilated_all_cases_pca.py
```

## 输入

- `data/final/dataset_manifest.csv`
- `data/final/datasets/工况*.csv`
- `data/added/dataset_manifest.csv`
- `data/added/standardized_datasets/工况21-24.csv`
- `data/added2/dataset_manifest.csv`
- `data/added2/standardized_datasets/工况25-30.csv`
- 复用窗口构造和训练工具：
  - `src/try/053_support_window_residual_quickcheck/run_support_window_residual_quickcheck.py`
  - `src/try/066_reuse_embedding_domain_split/reuse_embedding_domain_common.py`
  - `src/try/071_external_embedding_regression_quickcheck/run_external_embedding_regression_quickcheck.py`

## 输出

- 输出目录：`outputs/try/089_noncausal_cnn_encoder_ablation/`
- 每个 encoder 变体输出：
  - `embedding_case_table.csv`
  - `all_case_predictions.csv`
  - `summary_by_protocol.csv`
  - `summary_by_protocol_and_domain.csv`
  - `summary.md`
  - `models/checkpoints/*.pt`
  - `models/checkpoints/*_norm.npz`
  - `models/checkpoints/*.json`
- 总表：
  - `combined_summary_by_protocol.csv`
  - `combined_summary_by_protocol_and_domain.csv`
  - `summary.md`
- `noncausal_dilated` PCA 输出：
  - `noncausal_dilated/all_case_embedding_pca_coords.csv`
  - `noncausal_dilated/all_case_embedding_pca_summary.md`
  - `noncausal_dilated/plots/all_case_embedding_pca.png`
  - `noncausal_dilated/plots/external_case_embedding_pca.png`

## 说明

- `noncausal_dilated` 使用 dilation `1, 2, 4`，用于对照“只去 causal crop，保留 dilation 感受野”。
- `noncausal_nodilation` 使用 dilation `1, 1, 1`，用于进一步隔离 dilation / 感受野贡献。
- 训练数据沿用 `071` embedding 资产的 added-first 口径：`final` 带标签工况 `1, 3-20` + `added` 带标签工况 `21-24`。
- 测试口径沿用 `071`：`added(21-24) -> added2(25-30)`，并保留 external LOOCV 作为次级复核。
