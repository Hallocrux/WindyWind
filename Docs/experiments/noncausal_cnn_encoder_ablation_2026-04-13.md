# non-causal CNN encoder ablation

- 状态：`current`
- 首次确认：`2026-04-13`
- 最近复核：`2026-04-13`
- 数据范围：
  - encoder 训练：`final` 带标签工况 `1, 3-20` + `added` 带标签工况 `21-24`
  - residual 训练：`data/added/` 的带标签工况 `21-24`
  - 测试：`data/added2/` 的带标签工况 `25-30`
  - 次级复核：外部带标签工况 `21-30`
- 代码口径：
  - `src/try/089_noncausal_cnn_encoder_ablation/`
- 证据入口：
  - `outputs/try/089_noncausal_cnn_encoder_ablation/combined_summary_by_protocol.csv`
  - `outputs/try/089_noncausal_cnn_encoder_ablation/noncausal_dilated/all_case_predictions.csv`
  - `outputs/try/089_noncausal_cnn_encoder_ablation/noncausal_dilated/all_case_embedding_pca_coords.csv`
  - `outputs/try/089_noncausal_cnn_encoder_ablation/noncausal_dilated/all_case_embedding_pca_summary.md`
  - `outputs/try/089_noncausal_cnn_encoder_ablation/noncausal_dilated/plots/all_case_embedding_pca.png`
  - `outputs/try/089_noncausal_cnn_encoder_ablation/noncausal_nodilation/all_case_predictions.csv`
  - `outputs/try/089_noncausal_cnn_encoder_ablation/summary.md`
- 模型资产：
  - `outputs/try/089_noncausal_cnn_encoder_ablation/noncausal_dilated/models/checkpoints/noncausal_dilated_2s.pt`
  - `outputs/try/089_noncausal_cnn_encoder_ablation/noncausal_dilated/models/checkpoints/noncausal_dilated_2s_norm.npz`
  - `outputs/try/089_noncausal_cnn_encoder_ablation/noncausal_dilated/models/checkpoints/noncausal_dilated_2s.json`
  - `outputs/try/089_noncausal_cnn_encoder_ablation/noncausal_dilated/models/checkpoints/noncausal_dilated_8s.pt`
  - `outputs/try/089_noncausal_cnn_encoder_ablation/noncausal_dilated/models/checkpoints/noncausal_dilated_8s_norm.npz`
  - `outputs/try/089_noncausal_cnn_encoder_ablation/noncausal_dilated/models/checkpoints/noncausal_dilated_8s.json`
  - `outputs/try/089_noncausal_cnn_encoder_ablation/noncausal_nodilation/models/checkpoints/noncausal_nodilation_2s.pt`
  - `outputs/try/089_noncausal_cnn_encoder_ablation/noncausal_nodilation/models/checkpoints/noncausal_nodilation_2s_norm.npz`
  - `outputs/try/089_noncausal_cnn_encoder_ablation/noncausal_nodilation/models/checkpoints/noncausal_nodilation_2s.json`
  - `outputs/try/089_noncausal_cnn_encoder_ablation/noncausal_nodilation/models/checkpoints/noncausal_nodilation_8s.pt`
  - `outputs/try/089_noncausal_cnn_encoder_ablation/noncausal_nodilation/models/checkpoints/noncausal_nodilation_8s_norm.npz`
  - `outputs/try/089_noncausal_cnn_encoder_ablation/noncausal_nodilation/models/checkpoints/noncausal_nodilation_8s.json`

## 目标

本探索基于 `071` 的下游口径，只做 encoder 归纳偏置消融：

- 保留 residual block；
- 保留每个 block 两层 `Conv1d`；
- 保留全局平均池化；
- 保持每个窗口 encoder embedding 维度为 `32`；
- 保持 `2s + 8s` 拼接后的 case embedding 维度为 `64`；
- 保持 `071` 下游 `rpm_knn4 + embedding residual ridge` 接口兼容；
- 将原 TinyTCN 中的 causal crop 去掉，改成 non-causal 对称 padding；
- 对比 `with_dilation=True` 与 `with_dilation=False`。

两个变体：

- `noncausal_dilated`：dilation 为 `1, 2, 4`；
- `noncausal_nodilation`：dilation 为 `1, 1, 1`。

## 运行方式

```powershell
uv run python src/try/089_noncausal_cnn_encoder_ablation/run_noncausal_cnn_encoder_ablation.py
```

## 主结果

`2026-04-13` 的 `added_to_added2` 结果：

| encoder | variant | case_mae | case_rmse | mean_signed_error |
| --- | --- | ---: | ---: | ---: |
| `noncausal_dilated` | `rpm_knn4_plus_embedding_residual_ridge` | `0.5270` | `0.7199` | `-0.2242` |
| `071 causal TinyTCN` | `rpm_knn4_plus_embedding_residual_ridge` | `0.6161` | `0.7621` | `-0.1427` |
| `noncausal_nodilation` | `rpm_knn4_plus_embedding_residual_ridge` | `1.2563` | `1.4232` | `0.9150` |
| `rpm_knn4` | 无 residual | `1.2903` | `1.5511` | `-0.6364` |
| `noncausal_dilated` | `embedding_ridge` | `1.3574` | `1.5404` | `-0.4473` |
| `noncausal_nodilation` | `embedding_ridge` | `1.4604` | `1.6437` | `0.6752` |

`noncausal_dilated` 的 residual ridge 在主测试上比 `071` 的 `0.6161` 改善到 `0.5270`。`noncausal_nodilation` 只略优于 `rpm_knn4`，没有形成稳定增益。

## 逐工况表现

`2026-04-13` 的 `noncausal_dilated | rpm_knn4_plus_embedding_residual_ridge` 在 `added2` 各工况表现：

| 工况 | true | pred | abs_error |
| --- | ---: | ---: | ---: |
| `25` | `8.5000` | `6.9117` | `1.5883` |
| `26` | `7.7000` | `7.4051` | `0.2949` |
| `27` | `6.5000` | `6.6901` | `0.1901` |
| `28` | `3.7000` | `3.3294` | `0.3706` |
| `29` | `3.6000` | `3.7741` | `0.1741` |
| `30` | `3.3000` | `3.8443` | `0.5443` |

相对 `071` 的逐工况误差：

| 工况 | `noncausal_dilated` abs_error | `071` abs_error | 变化 |
| --- | ---: | ---: | ---: |
| `25` | `1.5883` | `1.2068` | 变差 `0.3815` |
| `26` | `0.2949` | `0.0909` | 变差 `0.2040` |
| `27` | `0.1901` | `0.6505` | 改善 `0.4604` |
| `28` | `0.3706` | `1.0697` | 改善 `0.6991` |
| `29` | `0.1741` | `0.0059` | 变差 `0.1682` |
| `30` | `0.5443` | `0.6729` | 改善 `0.1286` |

主测试整体改善主要来自 `工况27`、`工况28`、`工况30`，其中 `工况28` 的改善最大。

## 次级复核

`2026-04-13` 的 `external_loocv` 结果：

| encoder | variant | case_mae | case_rmse |
| --- | --- | ---: | ---: |
| 无 embedding | `rpm_knn4` | `0.7772` | `1.0102` |
| `noncausal_dilated` | `rpm_knn4_plus_embedding_residual_ridge` | `0.8426` | `1.0830` |
| `noncausal_nodilation` | `rpm_knn4_plus_embedding_residual_ridge` | `0.9033` | `1.1829` |

这说明 `noncausal_dilated` 在主测试上超过 `071`，但 external LOOCV 仍没有超过纯 `rpm_knn4`；它应作为强候选进入后续复核，而不是立即替换为最终默认模型。

## 全部工况 PCA

`2026-04-13` 使用 `noncausal_dilated` 的全部 `30` 个工况、`64` 维 case embedding 做标准化 PCA：

- 输出坐标：`outputs/try/089_noncausal_cnn_encoder_ablation/noncausal_dilated/all_case_embedding_pca_coords.csv`
- 输出图：`outputs/try/089_noncausal_cnn_encoder_ablation/noncausal_dilated/plots/all_case_embedding_pca.png`
- PCA explained variance：
  - `PC1 = 59.60%`
  - `PC2 = 26.13%`

## 结论

`2026-04-13` 的消融结果支持以下判断：

- 对完整时序窗口编码任务，causal convolution 不是必要归纳偏置；
- 去掉 causal crop 并保留 dilation 后，`added_to_added2` residual ridge 从 `071` 的 `0.6161` 改善到 `0.5270`；
- 去掉 dilation 后，`case_mae = 1.2563`，几乎退回到 `rpm_knn4 = 1.2903` 附近；
- 因此本轮收益更可能来自“non-causal 对称上下文 + dilation 感受野 + residual block”的组合，而不是 causal 结构本身；
- 由于 `external_loocv` 未超过 `rpm_knn4`，`2026-04-13` 不直接替换最终默认模型，但 `noncausal_dilated` 是下一步 added-first 复核的优先候选。
