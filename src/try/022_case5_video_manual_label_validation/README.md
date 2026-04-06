# 022 工况5视频手工标注定向验证

## 目标

- 在一个新的 try 中，围绕 `2026-04-06` 已确认的事实重做视频手工标注验证：
  - `data/video/VID_20260330_162635.mp4` 就是 `工况5` 实录；
  - `outputs/annotations/test/*` 是该视频片段的手工标注与视频侧 RPM 证据。
- 复用当前 `TinyTCN` 转速回归口径，做“工况5定向验证”，而不是再把问题改写成 pre-case5 子验证。
- 当前手工标注片段为 `51` 帧，约 `0.8390s`，明显短于表格侧现有模型使用的 `5s` 窗。

## 输入与口径

- 手工标注证据：
  - `config/test.yaml`
  - `outputs/annotations/test/summary.json`
  - `outputs/annotations/test/video_rpm_eval.json`
  - 当前片段为 `51` 帧，约 `0.8390s`
- 表格数据：
  - `data/final/dataset_manifest.csv`
  - `data/final/datasets/工况5.csv`
  - `outputs/dataset_inventory.csv`
- 表格清洗：复用 `src/current.data_loading`
- 原始窗口构造：复用 `src/try/012_phase3_end_to_end_shortlist/`
- TinyTCN rpm 训练：复用 `src/try/019_tinytcn_rpm_regression/`
- 默认窗口参数：
  - 采样率 `50Hz`
  - 窗长 `5s`
  - 步长 `2.5s`

## 验证问题

- 视频片段手工 RPM 是多少；
- 同片段的视频频谱 RPM 是多少；
- `工况5` manifest rpm 是多少；
- 以“除工况5外的全部带 rpm 标签工况”为训练集时，`TinyTCN` 对 `工况5` 的 holdout 预测是多少；
- 上述几项之间的差值分别是多少。
- 在缺少显式视频-表格同步信号时，如果先采用“视频起点约对齐工况5表格起点”的工作假设，最接近手工片段中心时刻的现有 `5s` 窗预测是多少。

## 运行方式

```powershell
uv run python src/try/022_case5_video_manual_label_validation/run_case5_video_manual_label_validation.py
```

## 输出

- 输出目录：`outputs/try/022_case5_video_manual_label_validation/`
- 固定产物：
  - `manual_reference_summary.csv`
  - `case5_evidence_summary.csv`
  - `case5_loco_window_predictions.csv`
  - `case5_comparison_summary.csv`
  - `summary.md`

## 当前用途

- 这个 try 用于验证“工况5视频片段手工标注”与“工况5表格/模型结果”是否一致；
- 它不是为了替代 manifest rpm，而是为了定位视频侧证据与表格侧标签之间是否存在值得继续追查的差异。
- 对“51 帧片段 -> 5s 表格窗”的映射，当前只采用工作假设，不应误读成已经完成严格同步。
