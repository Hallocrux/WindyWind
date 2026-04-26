# 085 added+added2 训练池的 2s+8s 晚融合 LOCO quickcheck

## 目标

- 以 `data/added/` 与 `data/added2/` 的带标签工况作为统一训练池；
- 复用现有 `TinyTCN@2s`、`TinyTCN@8s` 与工况级 `2s+8s` 晚融合；
- 只挑少量代表性工况做 case-level `LOCO`，快速检查：
  - 在 `added + added2` 内部训练时，direct learned `TinyTCN 2s+8s` 是否有可保留信号；
  - `2s`、`8s`、`2s+8s` 三者在这个小池子里谁更稳。

## 默认 holdout

- `工况22`：`added` 低风速 hard case
- `工况23`：`added` 高风速代表
- `工况25`：`added2` 高风速代表
- `工况29`：`added2` 低风速代表

如需改成别的少量工况，可用 `--holdout-case-ids` 覆盖。

## 输入

- `data/added/dataset_manifest.csv`
- `data/added/standardized_datasets/工况21-24.csv`
- `data/added2/dataset_manifest.csv`
- `data/added2/standardized_datasets/工况25-30.csv`

## 运行

```powershell
uv run python src/try/085_added_added2_late_fusion_loco_quickcheck/run_added_added2_late_fusion_loco_quickcheck.py
```

自定义 holdout：

```powershell
uv run python src/try/085_added_added2_late_fusion_loco_quickcheck/run_added_added2_late_fusion_loco_quickcheck.py --holdout-case-ids 22 23 25 29
```

## 输出

- 输出目录：`outputs/try/085_added_added2_late_fusion_loco_quickcheck/`
- 固定产物：
  - `case_level_predictions.csv`
  - `summary_by_domain.csv`
  - `summary.md`
  - `run_config.json`
