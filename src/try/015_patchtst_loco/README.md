# 015 PatchTST LOCO

- 状态：`current`
- 首次确认：`2026-04-05`
- 最近复核：`2026-04-05`

## 目标

在 windywind 的 `50Hz / 5s / 2.5s` 原始窗口口径上，复用官方 `PatchTST` 仓库骨干，做按工况 `Leave-One-Condition-Out` 的窗口级风速回归。

## 官方代码来源

- 仓库：`https://github.com/yuqinie98/PatchTST`
- commit：`204c21efe0b39603ad6e2ca640ef5896646ab1a9`
- 本地镜像：
  - `src/try/015_patchtst_loco/vendor_PatchTST/`

## 输入

- `data/final/dataset_manifest.csv`
- `data/final/datasets/工况*.csv`

## 运行方式

开发集 smoke：

```bash
uv run python src/try/015_patchtst_loco/run_patchtst_loco.py --mode dev
```

全量工况：

```bash
uv run python src/try/015_patchtst_loco/run_patchtst_loco.py --mode full
```

## 当前适配口径

- `2026-04-05` 直接复用官方 `PatchTST_supervised/models/PatchTST.py` 主体。
- `2026-04-05` 由于 windywind 任务是标量回归，不是原论文的多步同维度预测，因此只在官方骨干输出之后追加一个标量读出头，不重写 Transformer 主体。

## 输出位置

- `outputs/try/015_patchtst_loco/model_summary.csv`
- `outputs/try/015_patchtst_loco/case_level_predictions.csv`
- `outputs/try/015_patchtst_loco/unlabeled_predictions.csv`
- `outputs/try/015_patchtst_loco/summary.md`
- `outputs/try/015_patchtst_loco/run_config.json`
