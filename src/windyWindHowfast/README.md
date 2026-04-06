# windyWindHowfast

视频 ROI -> RPM 分析子模块。

## 当前说明（2026-04-06）

- `data/video/VID_20260330_162635.mp4` 当前按项目口径视为 `工况5` 实录；
- 这个子模块是当前 `工况5` 视频的 RPM CV 主线；
- 与该视频配套的手工标注、RPM 拟合和逐帧 ROI 验证位于 `src/windNotFound/`。

## 目标

- 保留角度维度上的结构信息，不再把整圈角度平均成单一强度信号。
- 在极坐标空间中，把叶片旋转视为“角度轴上的平移”。
- 通过时间频率 `f` 与角向模态 `k` 的二维谱峰，反推出转子频率 `f / k`。

## 当前结构

- 当前对外只保留视频主链入口：
  - `main.py`
  - `roi_detection.py`
  - `analysis_core.py`
  - `support.py`
- 自动 ROI 内部仍采用三层机制：
  - 候选生成 `candidate generators`
  - 统一评分 `candidate scoring`
  - 选择与回退 `selection + fallback`
- 当前默认启用两类 generator：
  - `motion`
  - `static_structure`
- 后续可以继续接入：
  - `detector`
  - `keypoint`
  - `segmentation`

## 运行

默认自动 ROI：

```bash
uv run python -m src.windyWindHowfast --video data/video/VID_20260330_162635.mp4
```

显式指定 ROI 时，会跳过自动候选框架：

```bash
uv run python -m src.windyWindHowfast \
  --video data/video/VID_20260330_162635.mp4 \
  --roi-algorithm manual \
  --center-x 455 \
  --center-y 800 \
  --radius 230
```

手工标注与 21 点 RPM 拟合已迁移到 `src/windNotFound/`。

## 关键 CLI 参数

- `--roi-algorithm`
- `--roi-frame-strategy`
- `--roi-reference-max-frames`
- `--roi-score-threshold`
- `--roi-debug`
- `--roi-json`
- `--center-x --center-y --radius`

## 输出工件

每次运行默认写到 `outputs/windyWindHowfast/<视频名>/`：

- `<run-name>_roi_candidates.json`
- `<run-name>_roi_detection.json`
- `<run-name>_roi_best_debug.png`
- `<run-name>_roi_candidate_<idx>.png`
- `<run-name>_analysis_result.json`
- `<run-name>_analysis_summary.png`
- `<run-name>_first_frame_with_roi.png`
- `<run-name>_roi.json`

## 当前定位

- 当前版本是“弱泛化、可扩展”的 ROI 候选框架。
- 现在主要在仓库已有的室内风机视频上优先测试。
- 当前重点是把 ROI 获取阶段做成可扩展基础设施，而不是把检测逻辑写死成某个样本的固定规则。

## 相关证据（2026-04-06）

- `config/test.yaml`
- `outputs/windyWindHowfast/VID_20260330_162635/`
- `outputs/annotations/test/summary.json`
- `outputs/annotations/test/video_rpm_eval.json`

## 文档入口（2026-04-06）

- 顶层总览：`Docs/video_pipeline.md`
- 当前子项目细节：`Docs/video_rpm_cv_pipeline.md`
- 手工标注资产与验证链：`Docs/video_manual_annotation_assets.md`
- 详细文档：`Docs/video_rpm_cv_pipeline.md`
