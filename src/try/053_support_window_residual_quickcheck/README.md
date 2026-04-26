# 053 support-window residual quickcheck

## 目标

- 在 `052` 已经给出 `2s+8s` case-level embedding residual 正信号的前提下，继续验证：
  - 先用 `rpm_knn4` 找参考工况；
  - 再把参考工况的窗口时序作为 support bank；
  - 用窗口级 embedding 匹配来做 residual 修正；
- 控制成本：
  - 只测代表性 holdout；
  - 旧对照直接复用 `052` 已有输出；
  - 新训练模型做 checkpoint 持久化，下次复跑直接复用。

## holdout 工况

- `工况1`
- `工况3`
- `工况17`
- `工况18`
- `工况21`
- `工况22`
- `工况23`
- `工况24`

## 方法摘要

- `base`：
  - `rpm_knn4`
- `reference cases`：
  - 由 `rpm_knn4` 的 `k=4` 邻居确定
- `support-window residual`：
  - 对每个 holdout 窗口，在参考工况的窗口 embedding 中找近邻；
  - 邻居窗口携带所属参考工况的 `rpm residual oof`；
  - 窗口级 residual 做距离加权平均，再聚成 case-level correction。

## 运行方式

```powershell
uv run python src/try/053_support_window_residual_quickcheck/run_support_window_residual_quickcheck.py
```

## 输出

- 输出目录：`outputs/try/053_support_window_residual_quickcheck/`
- 主要文件：
  - `case_level_predictions.csv`
  - `summary_by_domain.csv`
  - `support_window_neighbors.csv`
  - `models/checkpoints/`
  - `summary.md`
