# 018 结构基频候选扫描

## 目标

- 在 `2026-04-05` 当前数据口径下，为“结构基频识别”补一个可复现的第一阶段探索入口。
- 复用正式主线的清洗逻辑，先回答两个问题：
  - 去掉转频及其低阶倍频干扰后，时序里是否还存在稳定的候选结构频率；
  - 候选频率是否在多数工况上集中到相近频带，从而可以作为后续正式任务的优先方向。

## 输入数据

- `data/final/dataset_manifest.csv`
- `data/final/datasets/工况{ID}.csv`

## 方法口径

- 代码口径：
  - `src/current/data_loading.py`
- 默认窗口口径：
  - 采样率：`50 Hz`
  - 窗长：`250` 点，即 `5s`
  - 步长：`125` 点，即 `2.5s`
- 当前只对共有有效通道中的两类信号做候选扫描：
  - `应变传感器*.chdata`
  - `WSMS*.Acc*`
- 对每个窗口做单边频谱，并在 `0.5Hz - 8.0Hz` 范围内寻找候选主峰。
- 若工况带 `rpm` 标签，则默认屏蔽：
  - `1x` 到 `4x` 转频
  - 每个倍频点附近默认 `±0.2Hz`

## 运行方式

```powershell
uv run python src/try/018_structural_fundamental_frequency_scan/run_structural_frequency_scan.py
```

也可以显式调整搜索带宽：

```powershell
uv run python src/try/018_structural_fundamental_frequency_scan/run_structural_frequency_scan.py `
  --freq-min 0.5 `
  --freq-max 8.0 `
  --rotor-exclusion-width 0.2
```

## 输出位置

- `outputs/try/018_structural_fundamental_frequency_scan/case_frequency_summary.csv`
- `outputs/try/018_structural_fundamental_frequency_scan/window_frequency_candidates.csv`
- `outputs/try/018_structural_fundamental_frequency_scan/structural_frequency_overview.png`

## 输出解读

- `case_frequency_summary.csv`
  - 每个工况的转频、应变候选频率中位数、加速度候选频率中位数、候选窗口数。
- `window_frequency_candidates.csv`
  - 每个窗口、每类传感器组的候选频率，便于后续做稳定性分析或伪标签构造。
- `structural_frequency_overview.png`
  - 左图：各工况候选频率与转频对比；
  - 右图：应变候选频率分布直方图。

## 当前用途

- 这个探索入口服务于“结构基频识别”的第一阶段，不是最终模型。
- 如果应变侧候选频率在多数工况上持续集中到同一窄频带，下一步优先做：
  - 稳态窗口筛选；
  - 多传感器一致性评分；
  - 结构基频候选的正式基线提取器。
