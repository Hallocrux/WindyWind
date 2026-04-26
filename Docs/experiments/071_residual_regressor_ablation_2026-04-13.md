# 071 residual regressor ablation

- 状态：`current`
- 首次确认：`2026-04-13`
- 最近复核：`2026-04-13`
- 数据范围：
  - residual 训练：`data/added/` 的带标签工况 `21-24`
  - 测试：`data/added2/` 的带标签工况 `25-30`
  - 次级复核：外部带标签工况 `21-30`
- 代码口径：
  - `src/try/088_071_residual_regressor_ablation/`
- 证据入口：
  - `outputs/try/088_071_residual_regressor_ablation/summary_by_protocol.csv`
  - `outputs/try/088_071_residual_regressor_ablation/added_to_added2_compare_vs_071.csv`
  - `outputs/try/088_071_residual_regressor_ablation/all_case_predictions.csv`
  - `outputs/try/088_071_residual_regressor_ablation/summary.md`

## 目标

本探索固定 `071` 的主要口径，只替换 residual regressor：

- embedding 表：`outputs/try/069_added2_embedding_pca_projection/embedding_case_table.csv`
- 主干：`rpm_knn4`
- residual target：训练子集内部 `rpm_knn4` 近似 OOF residual
- 主协议：`added(21-24) -> added2(25-30)`

问题是：`071 | rpm_knn4 + embedding residual ridge` 的瓶颈是否主要来自 Ridge 回归器本身。

## 运行方式

```powershell
uv run python src/try/088_071_residual_regressor_ablation/run_071_residual_regressor_ablation.py
```

## 主结果

`2026-04-13` 的 `added_to_added2` 结果：

| variant | case_mae | delta vs 071 | 结论 |
| --- | ---: | ---: | --- |
| `rpm_knn4_plus_residual_svr_linear` | `0.6089` | `-0.0072` | 主测试略优于 071 |
| `rpm_knn4_plus_residual_ridge_cv` | `0.6161` | `0.0000` | 071 对照 |
| `rpm_knn4_plus_residual_pls3` | `0.6161` | `0.0000` | 几乎等同 071 |
| `rpm_knn4_plus_residual_kernel_ridge_linear` | `0.6318` | `0.0156` | 弱于 071 |
| `rpm_knn4_plus_residual_elasticnet_cv` | `0.6359` | `0.0198` | 弱于 071 |
| `rpm_knn4_plus_residual_lasso_cv` | `0.7248` | `0.1086` | 弱于 071 |
| `rpm_knn4_plus_residual_pls2` | `0.7354` | `0.1193` | 弱于 071 |
| `rpm_knn4_plus_residual_pls1` | `0.9012` | `0.2851` | 弱于 071 |
| `rpm_knn4_plus_residual_extra_trees` | `0.9624` | `0.3463` | 弱于 071 |
| `rpm_knn4_plus_residual_knn1` | `1.0538` | `0.4377` | 弱于 071 |
| `rpm_knn4_plus_residual_knn2` | `1.0710` | `0.4549` | 弱于 071 |
| `rpm_knn4_plus_residual_random_forest` | `1.0749` | `0.4588` | 弱于 071 |
| `rpm_knn4_plus_residual_kernel_ridge_rbf` | `1.1350` | `0.5189` | 弱于 071 |
| `rpm_knn4_plus_residual_svr_rbf` | `1.1459` | `0.5298` | 弱于 071 |
| `rpm_knn4_plus_residual_knn4` | `1.1771` | `0.5610` | 弱于 071 |
| `rpm_knn4` | `1.2903` | `0.6742` | 无 residual 对照 |

## 逐工况对比

`2026-04-13` 主测试中，linear SVR 相对 071 ridge 的变化：

| 工况 | true | linear SVR pred | linear SVR abs error | 071 ridge pred | 071 ridge abs error |
| --- | ---: | ---: | ---: | ---: | ---: |
| `25` | `8.5000` | `7.1696` | `1.3304` | `7.2932` | `1.2068` |
| `26` | `7.7000` | `7.6623` | `0.0377` | `7.7909` | `0.0909` |
| `27` | `6.5000` | `7.0238` | `0.5238` | `7.1505` | `0.6505` |
| `28` | `3.7000` | `2.6144` | `1.0856` | `2.6303` | `1.0697` |
| `29` | `3.6000` | `3.5107` | `0.0893` | `3.6059` | `0.0059` |
| `30` | `3.3000` | `3.8866` | `0.5866` | `3.9729` | `0.6729` |

linear SVR 的整体优势来自 `工况26`、`工况27`、`工况30`，但在 `工况25`、`工况28`、`工况29` 弱于 071 ridge。

## 次级复核

`2026-04-13` 的 `external_loocv` 结果显示：

- `rpm_knn4_plus_residual_random_forest`
  - `case_mae = 0.7641`
- `rpm_knn4`
  - `case_mae = 0.7772`
- `rpm_knn4_plus_residual_ridge_cv`
  - `case_mae = 0.8451`
- `rpm_knn4_plus_residual_svr_linear`
  - `case_mae = 2.6684`

这说明 linear SVR 在主测试中的微弱优势没有跨协议稳定性。

## 结论

`2026-04-13` 的同口径替换实验显示，Ridge 不是明显瓶颈：

- linear SVR 在 `added_to_added2` 主测试上以 `0.6089` 略优于 071 ridge 的 `0.6161`，差距只有 `0.0072`；
- linear SVR 的 `external_loocv` 表现显著恶化到 `2.6684`，不适合作为默认替换；
- `PLS3` 与 ridge 几乎等价；
- kernel ridge linear、ElasticNet、Lasso、PLS1/2、kNN、tree ensemble、RBF SVR/RBF kernel ridge 在主测试中都未超过 071 ridge；
- `2026-04-13` 起，默认 added-first 最佳模型仍应保持 `071 | rpm_knn4 + embedding residual ridge`。
