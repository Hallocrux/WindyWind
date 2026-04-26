# 054 case evidence aggregator quickcheck

## 目标

- 不再给窗口直接分配 case-level 常数 residual；
- 只在 case 层使用已有候选修正与机制特征，预测：
  - `residual = true_wind_speed - rpm_knn4_base`
- 验证“窗口给证据，case 做决策”是否比直接套用某个固定 correction 更稳。

## 设计原则

- 只做代表性 holdout quickcheck；
- 尽量复用已有产物：
  - `052` 的 case-level residual 候选
  - `053` 的 support-window residual 候选
  - `047` 的 gate feature 表
- 新训练的小模型全部持久化，下次直接复用。

## holdout 工况

- `工况1`
- `工况3`
- `工况17`
- `工况18`
- `工况21`
- `工况22`
- `工况23`
- `工况24`

## 运行方式

```powershell
uv run python src/try/054_case_evidence_aggregator_quickcheck/run_case_evidence_aggregator_quickcheck.py
```

## 输出

- 输出目录：`outputs/try/054_case_evidence_aggregator_quickcheck/`
- 主要文件：
  - `evidence_table.csv`
  - `case_level_predictions.csv`
  - `summary_by_domain.csv`
  - `models/`
  - `summary.md`
