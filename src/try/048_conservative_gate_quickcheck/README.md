# 048 conservative gate quickcheck

## 目标

- 在 `047` 已验证 soft gate 有正信号但仍会伤害 `final` 的前提下，继续验证更保守的 gate 设计；
- 不再直接自由回归 `g in [0, 1]`；
- 优先比较：
  - 二分类 gate
  - 离散档位 gate
  - 两阶段 gate

## 专家定义

- `base expert`
  - `TinyTCN_multiscale_late_fusion_2s_8s`
- `enhanced expert`
  - `true_rpm -> rpm_knn4 + TinyTCN all_channels midband(3.0-6.0Hz) @ w=0.3`

## gate 特征

- 继续复用 `047` 的 `case-level` 特征；
- 不使用原始波形。

## 运行方式

```powershell
uv run python src/try/048_conservative_gate_quickcheck/run_conservative_gate_quickcheck.py
```

## 输出

- 输出目录：`outputs/try/048_conservative_gate_quickcheck/`
- 主要文件：
  - `dataset_table.csv`
  - `case_level_predictions.csv`
  - `summary_by_variant.csv`
  - `summary_by_domain.csv`
  - `summary.md`

## 说明

- 统一使用 `23` 个带标签工况的 case-level `LOOCV`；
- 对照组固定包括：
  - `base_only`
  - `enhanced_only`
  - `047_hgb_gate`
  - `global_weight_cv`
  - `oracle_soft_gate`
