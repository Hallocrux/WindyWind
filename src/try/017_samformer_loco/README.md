# 017 SAMformer LOCO

- 状态：`current`
- 首次确认：`2026-04-05`
- 最近复核：`2026-04-05`

## 目标

在 windywind 的 `50Hz / 5s / 2.5s` 原始窗口口径上，复用官方 `SAMformer` 的 PyTorch 版本骨干，做按工况 `Leave-One-Condition-Out` 的窗口级风速回归。

## 官方代码来源

- 仓库：`https://github.com/romilbert/samformer`
- commit：`71f10eaa696f2a098798779ee14b6ecd6b69bcd9`
- 本地镜像：
  - `src/try/017_samformer_loco/vendor_samformer/`

## 输入

- `data/final/dataset_manifest.csv`
- `data/final/datasets/工况*.csv`

## 运行方式

开发集 smoke：

```bash
uv run python src/try/017_samformer_loco/run_samformer_loco.py --mode dev
```

全量工况：

```bash
uv run python src/try/017_samformer_loco/run_samformer_loco.py --mode full
```

## 当前适配口径

- `2026-04-05` 直接复用官方 `samformer_pytorch/samformer/samformer.py` 中的 `SAMFormerArchitecture` 主体。
- `2026-04-05` 由于 windywind 的目标是标量风速回归，当前适配在官方通道级输出后追加一个标量读出头，不重写通道注意力主体。

## 输出位置

- `outputs/try/017_samformer_loco/model_summary.csv`
- `outputs/try/017_samformer_loco/case_level_predictions.csv`
- `outputs/try/017_samformer_loco/unlabeled_predictions.csv`
- `outputs/try/017_samformer_loco/summary.md`
- `outputs/try/017_samformer_loco/run_config.json`
