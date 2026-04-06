# 021 TinyTCN 手工标注 pre-case5 验证

## 状态

- 状态：`failed`
- 首次确认：`2026-04-06`
- 最近复核：`2026-04-06`
- 替代关系：
  - 被 `src/try/022_case5_video_manual_label_validation/` 替代

## 失败原因

- `2026-04-06` 复核后已确认：
  - `data/video/VID_20260330_162635.mp4` 就是 `工况5` 实录；
  - 本 try 把这段视频误读成了“工况5之前”的外部参考。
- 因此本 try 的问题定义已经偏离用户原意：
  - 它错误地排除了 `工况5`；
  - 把验证目标转成了 `case_id <= 4` 的 pre-case5 子问题；
  - 结论不能继续用作“工况5视频手工标注验证”的有效依据。

## 目标

- 在一个新的 try 中，使用仓库里“工况5之前”的手工标注 RPM 参考，验证当前 `TinyTCN` 转速回归路线。
- 不改动任何原有文件，只在本 try 目录和对应输出目录内新增文件。

## 核心假设

- `2026-04-06` 仓库内能直接复用的已完成手工标注 RPM 证据，来自：
  - `outputs/annotations/test/summary.json`
  - `outputs/annotations/test/video_rpm_eval.json`
- 这组手工标注对应原始视频 `data/video/VID_20260330_162635.mp4` 上的一个早期 selector 片段。
- 仓库内没有显式的“视频帧区间 -> 表格工况编号”映射表，因此本 try 采用一个保守口径：
  - 把这组手工 RPM 当作“工况5之前”的外部参考；
  - 用 `case_id <= 4` 的表格工况构造一个 pre-case5 TinyTCN rpm 子验证；
  - 训练集使用其中带 rpm 标签的工况；
  - 推理目标使用其中唯一无 rpm 标签的 `工况2`；
  - 再把 `工况2` 的预测值与手工 RPM 参考做对比。

## 输入与口径

- 手工标注证据：
  - `outputs/annotations/test/summary.json`
  - `outputs/annotations/test/video_rpm_eval.json`
- 表格数据：
  - `data/final/dataset_manifest.csv`
  - `data/final/datasets/工况*.csv`
- 表格清洗：复用 `src/current.data_loading`
- 原始窗口构造：复用 `src/try/012_phase3_end_to_end_shortlist/`
- TinyTCN rpm 训练与推理：复用 `src/try/019_tinytcn_rpm_regression/`
- 默认窗口参数：
  - 采样率 `50Hz`
  - 窗长 `5s`
  - 步长 `2.5s`

## 运行方式

```powershell
uv run python src/try/021_tinytcn_manual_labels_before_case5_validation/run_manual_labels_before_case5_validation.py
```

## 输出

- 输出目录：`outputs/try/021_tinytcn_manual_labels_before_case5_validation/`
- 固定产物：
  - `manual_reference_summary.csv`
  - `pre_case5_candidate_case_rpm.csv`
  - `validation_summary.csv`
  - `summary.md`

## 限制

- `2026-04-06` 仓库里没有找到“手工标注片段”与“工况1-4/工况5”的官方对齐表。
- 因此这里的验证属于“基于现有证据的最接近用户意图验证”，不是已经完成工况级精确配准的严格真值评测。
