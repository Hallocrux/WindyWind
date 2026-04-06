# 视频主线说明

本文档记录视频 RPM 模块与手工标注子项目的当前工程状态。  
视频主线和表格主线彼此相关，但当前作为独立模块维护。

## 0. 文档入口

- 顶层摘要：
  - 本文档
- 视频 RPM CV 主线细节：
  - `Docs/video_rpm_cv_pipeline.md`
- 手工标注资产与验证链细节：
  - `Docs/video_manual_annotation_assets.md`

## 1. 视频 RPM 模块当前状态

- 状态：`current`
- 首次确认：`2026-04-03`
- 最近复核：`2026-04-06`

当前代码位置：

- `src/windyWindHowfast/`

当前默认样例视频：

- `data/video/VID_20260330_162635.mp4`

当前方法口径：

- 每帧先做风机 ROI 的极坐标展开
- 在极坐标图中沿半径维压缩，保留角度维
- 形成 `time_angle_map(t, theta)` 后做二维 FFT
- 显式排除 `k=0` 空间均匀项
- 由主导谱峰 `(f, k)` 反推转子频率

当前默认命令：

```bash
python -m src.windyWindHowfast
```

## 1.1 [2026-04-06] `VID_20260330_162635.mp4` 与 `工况5` 的当前对应关系

- 状态：`current`
- 首次确认：`2026-04-06`
- 最近复核：`2026-04-06`
- 数据范围：
  - `data/video/VID_20260330_162635.mp4`
  - `data/final/datasets/工况5.csv`
- 证据入口：
  - `config/test.yaml`
  - `outputs/annotations/test/summary.json`
  - `outputs/annotations/test/video_rpm_eval.json`
  - `outputs/dataset_inventory.csv`

当前按项目口径应明确记住：

- `VID_20260330_162635.mp4` 是 `工况5` 的实录视频；
- 该视频一直被用作当前 RPM CV 识别的主样本；
- `config/test.yaml` 与 `outputs/annotations/test/*` 是这段 `工况5` 视频片段的手工标注证据；
- 手工标注数据是当前视频侧的关键资产，原始证据应优先查看 `outputs/annotations/test/annotations.jsonl`；
- 后续若再出现“工况5之前视频”这类表述，应优先复核是否误读了这个映射。

## 2. ROI 机制与当前限制

- 状态：`current`
- 首次确认：`2026-04-03`
- 最近复核：`2026-04-06`
- 证据入口：
  - `outputs/windyWindHowfast/VID_20260330_162635/`
  - `src/try/002_auto_roi_failure_analysis/`

当前 ROI 框架：

- candidate generators
- candidate scoring
- selection + fallback

当前已实现 generator：

- `motion`
- `static_structure`

当前限制：

- 自动 ROI 在样例视频上仍可能选到叶片附近的局部高运动圆
- 因此当前视频 RPM 输出仍需结合 ROI 和频谱图一起判断

## 3. 手工标注子项目

- 状态：`current`
- 首次确认：`2026-04-03`
- 最近复核：`2026-04-06`

当前代码位置：

- `src/windNotFound/`

当前功能：

- 用 YAML 描述待标注帧
- 用 `jsonl` 追加写入标注结果
- 基于 `center -> blade_1` 角度拟合 selector 级 RPM
- 使用逐帧 ROI 对视频 RPM 算法做验证

当前固定规则：

- 支持的 selector：
  - `window`
  - `range`
  - `explicit`
- 单帧标注顺序：
  - `support_a`
  - `support_b`
  - `center`
  - `blade_1`
  - `blade_2`
  - `blade_3`
- `blade_1` 永远表示带 marker 的扇叶

## 3.1 [2026-04-06] 当前 `工况5` 视频的手工标注与视频验证证据

- 状态：`current`
- 首次确认：`2026-04-06`
- 最近复核：`2026-04-06`
- 数据范围：`VID_20260330_162635.mp4`
- 证据入口：
  - `config/test.yaml`
  - `outputs/annotations/test/annotations.jsonl`
  - `outputs/annotations/test/summary.json`
  - `outputs/annotations/test/video_rpm_eval.json`

当前可直接复用的 `工况5` 视频侧证据包括：

- selector：
  - `window`
  - 帧范围 `1975-2025`
  - 标注帧数 `51`
  - 时间跨度约 `0.8390s`
- 手工拟合 RPM：
  - `147.4341`
- 逐帧 ROI 视频频谱 RPM：
  - `140.2224`

这组结果的定位是：

- 它提供的是 `工况5` 视频片段上的视频侧 RPM 参考；
- 它不是表格 manifest 中 `166 rpm` 的自动替代物；
- 如果视频侧与 manifest 存在差异，应优先复核“视频片段覆盖时段、工况稳定性、以及视频-表格同步关系”。

## 4. 当前相关结论

- 状态：`historical`
- 首次确认：`2026-04-02`
- 最近复核：`2026-04-03`
- 数据范围：`VID_20260330_162635.mp4`
- 证据入口：
  - `outputs/windyWindHowfast/VID_20260330_162635/`
  - `src/try/002_auto_roi_failure_analysis/`

当前仍值得记住的结论：

- 自动 ROI 与人工参考 ROI 的偏差在样例视频上较大
- 不能把单次视频 RPM 数值直接视为稳定真值

## 5. 当前探索入口

- 状态：`current`
- 首次确认：`2026-04-06`
- 最近复核：`2026-04-06`

当前与 `工况5` 视频最相关的探索入口包括：

- `src/try/002_auto_roi_failure_analysis/`
  - 自动 ROI 为什么会在 `VID_20260330_162635.mp4` 上选错区域
- `src/try/005_case5_rpm_frequency_scan/`
  - `工况5.csv` 的频域诊断
- `src/try/006_case5_middle_segment_plots/`
  - `工况5` 中间非缺失部分逐列图
- `src/try/007_case5_core_middle_30s_plots/`
  - `工况5` 更稳中段 `30s` 逐列图
- `outputs/annotations/test/`
  - `工况5` 视频片段的手工标注与视频侧 RPM 验证
- `outputs/windyWindHowfast/VID_20260330_162635/`
  - `工况5` 视频的 CV / ROI / 频谱分析输出
- `Docs/video_rpm_cv_pipeline.md`
  - `windyWindHowfast` 详细主线说明
- `Docs/video_manual_annotation_assets.md`
  - 手工标注关键资产与验证链说明

## 6. 相关入口

- 当前项目状态总览：`PROJECT.md`
- 表格主线说明：`Docs/table_pipeline.md`
- 历史设计与重构说明：
  - `Docs/video_module_design_notes.md`
  - `Docs/howfast_refactor_final_spec_2026-04-03.md`
