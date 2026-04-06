# 016 MICN LOCO

- 状态：`current`
- 首次确认：`2026-04-05`
- 最近复核：`2026-04-05`

## 目标

在 windywind 的 `50Hz / 5s / 2.5s` 原始窗口口径上，复用官方 `MICN` 仓库骨干，做按工况 `Leave-One-Condition-Out` 的窗口级风速回归。

## 官方代码来源

- 仓库：`https://github.com/wanghq21/MICN`
- commit：`370c69b841d72246556ca05dd23163c560c22b5a`
- 本地镜像：
  - `src/try/016_micn_loco/vendor_MICN/`

## 输入

- `data/final/dataset_manifest.csv`
- `data/final/datasets/工况*.csv`

## 运行方式

开发集 smoke：

```bash
uv run python src/try/016_micn_loco/run_micn_loco.py --mode dev
```

全量工况：

```bash
uv run python src/try/016_micn_loco/run_micn_loco.py --mode full
```

## 当前适配口径

- `2026-04-05` 直接复用官方 `models/model.py` 中的 `MICN` 主体。
- `2026-04-05` windywind 不提供原论文格式的日历时间特征，因此当前适配用零填充 `x_mark_enc / x_mark_dec`，只保留官方网络主体与卷积结构。

## 输出位置

- `outputs/try/016_micn_loco/model_summary.csv`
- `outputs/try/016_micn_loco/case_level_predictions.csv`
- `outputs/try/016_micn_loco/unlabeled_predictions.csv`
- `outputs/try/016_micn_loco/summary.md`
- `outputs/try/016_micn_loco/run_config.json`
