# 视频 RPM CV 主线细节

- 状态：`current`
- 首次确认：`2026-04-06`
- 最近复核：`2026-04-06`
- 代码口径：`src/windyWindHowfast/`
- 数据范围：`data/video/VID_20260330_162635.mp4`
- 证据入口：
  - `src/windyWindHowfast/README.md`
  - `outputs/windyWindHowfast/VID_20260330_162635/`
  - `src/try/002_auto_roi_failure_analysis/`

## 1. 模块定位

这个文档只展开 `src/windyWindHowfast/` 的视频 RPM CV 主线细节。

当前角色：

- 负责从视频中自动或半自动获取风机 ROI；
- 在 ROI 内构造 `time_angle_map`；
- 用二维频谱方法估计 RPM；
- 为 `工况5` 实录视频提供当前主线的 CV / 频谱分析能力。

## 2. 当前方法口径

当前默认分析流程：

1. 从视频中选择或估计风机圆形 ROI；
2. 对 ROI 做极坐标展开；
3. 沿半径方向压缩，保留角度维，形成 `time_angle_map(t, theta)`；
4. 对 `time_angle_map` 做二维 FFT；
5. 在候选谱峰中反推出转子频率，再换算为 RPM。

当前方法重点：

- 显式排除 `k=0` 空间均匀项；
- 当前方法不是一维亮度时序 FFT，而是保留了角度结构；
- ROI 选择仍是当前误差和不稳定性的主要来源之一。

## 3. ROI 机制

当前 ROI 框架采用三层结构：

- candidate generators
- candidate scoring
- selection + fallback

当前默认 generator：

- `motion`
- `static_structure`

当前保留但未作为默认主线展开的方向：

- `detector`
- `keypoint`
- `segmentation`

## 4. 当前输入输出

默认输入：

- `data/video/VID_20260330_162635.mp4`

默认命令：

```bash
uv run python -m src.windyWindHowfast --video data/video/VID_20260330_162635.mp4
```

显式手工 ROI 命令：

```bash
uv run python -m src.windyWindHowfast \
  --video data/video/VID_20260330_162635.mp4 \
  --roi-algorithm manual \
  --center-x 455 \
  --center-y 800 \
  --radius 230
```

典型输出目录：

- `outputs/windyWindHowfast/VID_20260330_162635/`

典型工件：

- `*_roi_candidates.json`
- `*_roi_detection.json`
- `*_roi_best_debug.png`
- `*_analysis_result.json`
- `*_analysis_summary.png`
- `*_roi.json`

## 5. 当前已知限制

### 5.1 [2026-04-06] 自动 ROI 仍可能偏向局部高运动区域

- 状态：`current`
- 首次确认：`2026-04-03`
- 最近复核：`2026-04-06`

这意味着：

- 单次自动 ROI 输出不能直接当作稳定真值；
- 当前视频 RPM 结果仍需要结合 ROI 图和频谱图共同判断。

### 5.2 [2026-04-06] 当前主样本仍高度集中在工况5视频

- 状态：`current`
- 首次确认：`2026-04-06`
- 最近复核：`2026-04-06`

这意味着：

- 当前方法对 `工况5` 样本最熟悉；
- 对别的视频、别的机位或别的工况是否泛化，当前没有充分证据。

## 6. 相关探索链条

- `src/try/002_auto_roi_failure_analysis/`
  - 解释自动 ROI 为什么会在 `工况5` 视频上偏移
- `outputs/windyWindHowfast/VID_20260330_162635/`
  - 保存当前 CV 主线的实际分析输出
- `Docs/video_pipeline.md`
  - 上层总览文档
- `Docs/video_manual_annotation_assets.md`
  - 与手工标注子项目、关键标注资产和视频侧验证相关的详细文档
