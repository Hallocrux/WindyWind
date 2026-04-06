# 013 第三阶段 CNN / TCN smoke

## 目标

- 在一个独立 try 中，对第三阶段的两个轻量深度时序模型做最小可运行验证：
  - `Tiny1DCNN`
  - `TinyTCN`
- 默认只用小数据集，快速判断这条路线是否具备基本可用性。
- 保留一个手工特征参考：
  - `TabularReference_G6_Ridge`

## 输入与口径

- 数据入口：
  - `data/final/dataset_manifest.csv`
  - `data/final/datasets/工况*.csv`
- 清洗逻辑：复用 `src/current.data_loading`
- 原始窗口构造：复用 `src/try/012_phase3_end_to_end_shortlist/`
- 默认窗口参数：
  - 采样率 `50Hz`
  - 窗长 `5s`
  - 步长 `2.5s`
- 评估方式：
  - 分组单位为工况
  - `Leave-One-Condition-Out`

## 运行方式

开发态：

```powershell
uv run python src/try/013_phase3_cnn_tcn_smoke/run_cnn_tcn_smoke.py --mode dev
```

全量：

```powershell
uv run python src/try/013_phase3_cnn_tcn_smoke/run_cnn_tcn_smoke.py --mode full
```

显式指定工况：

```powershell
uv run python src/try/013_phase3_cnn_tcn_smoke/run_cnn_tcn_smoke.py --case-ids 1 2 3 5 15 16
```

## 输出

- 输出目录：`outputs/try/013_phase3_cnn_tcn_smoke/`
- 固定产物：
  - `cnn_tcn_model_summary.csv`
  - `cnn_tcn_case_level_predictions.csv`
  - `summary.md`

## 当前默认训练配置

- 设备：CPU
- batch size：`32`
- 最大 epoch：`40`
- early stopping patience：`8`
- 学习率：`1e-3`
- 目标：先看“是否可用”，不是追求当前最优分数
