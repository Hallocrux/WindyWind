# 012 第三阶段端到端 shortlist

## 目标

- 重新进入 `Docs/better_ds_spec_4_5.md` 的第三阶段：评估轻量原始时序模型。
- 当前环境没有 `torch / tensorflow / sktime / aeon`，因此本 try 先落地可直接运行的轻量原型，而不强行上 CNN / TCN。
- 本探索固定比较：
  - `TabularReference_G6_Ridge`
    - 当前阶段 1 shortlist 最优手工特征参考
  - `RawFlattenRidge`
    - 原始窗口直接展平后的线性参考
  - `MiniRocketLikeRidge`
    - 轻量随机卷积特征 + Ridge
  - `RawFlattenMLP`
    - 原始窗口直接展平后的轻量 MLP

## 输入与口径

- 数据入口：
  - `data/final/dataset_manifest.csv`
  - `data/final/datasets/工况*.csv`
- 清洗逻辑：复用 `src/current.data_loading`
- 切窗口径：
  - 采样率 `50Hz`
  - 窗长 `5s`
  - 步长 `2.5s`
- 评估方式：
  - 分组单位为工况
  - `Leave-One-Condition-Out`
- 默认优先看 `rpm-free`

## 运行方式

开发态：

```powershell
uv run python src/try/012_phase3_end_to_end_shortlist/run_phase3_shortlist.py --mode dev
```

全量：

```powershell
uv run python src/try/012_phase3_end_to_end_shortlist/run_phase3_shortlist.py --mode full
```

显式指定工况：

```powershell
uv run python src/try/012_phase3_end_to_end_shortlist/run_phase3_shortlist.py --case-ids 1 2 3 5 15 16
```

## 输出

- 输出目录：`outputs/try/012_phase3_end_to_end_shortlist/`
- 固定产物：
  - `phase3_model_summary.csv`
  - `phase3_case_level_predictions.csv`
  - `summary.md`

## 当前限制

- `2026-04-05` 本 try 不实现 CNN / TCN。
- `2026-04-05` 本 try 先判断“原始窗口路线是否有希望”，不是直接替换正式主线。
