# windNotFound

手工标注、21 点 RPM 直接拟合、逐帧 ROI 视频频谱验证子项目。

## 当前说明（2026-04-06）

- `config/test.yaml` 当前对应 `data/video/VID_20260330_162635.mp4`；
- `data/video/VID_20260330_162635.mp4` 当前按项目口径视为 `工况5` 实录；
- 这个子项目服务于 `工况5` 视频的手工标注、RPM 拟合和视频侧验证。

## 入口

- 标注：

```bash
uv run python src/windNotFound/run_annotate.py --task config/test.yaml
```

- 基于已有标注拟合 RPM：

```bash
uv run python src/windNotFound/run_fit_rpm.py --task config/test.yaml
```

- 用逐帧标注 ROI 验证视频频谱 RPM：

```bash
uv run python src/windNotFound/run_eval_video_rpm.py --task config/test.yaml --selector-index 0
```

## 输出

- `outputs/annotations/<task_stem>/annotations.jsonl`
- `outputs/annotations/<task_stem>/summary.json`
- `outputs/annotations/<task_stem>/video_rpm_eval.json`

`summary.json` 按 selector 输出：

- `rpm`
- `signed_rpm`
- `angular_velocity_rad_per_sec`
- `frame_span`
- `time_span_sec`
- `fit_residual_mae`

当前拟合口径只使用 `center -> blade_1` 角度，不做 `support_a/support_b` 稳像补偿。

视频频谱验证口径会对 selector 内每一帧应用该帧自己的标注 ROI，再做 `time-angle` 二维频谱。

## 当前相关证据（2026-04-06）

- `config/test.yaml`
- `outputs/annotations/test/annotations.jsonl`
- `outputs/annotations/test/summary.json`
- `outputs/annotations/test/video_rpm_eval.json`
- 详细文档：`Docs/video_manual_annotation_assets.md`

当前 `test` 任务对应：

- 视频：`data/video/VID_20260330_162635.mp4`
- selector：`window`
- 帧范围：`1975-2025`
- 标注帧数：`51`
- 时间跨度：约 `0.8390s`

## 文档入口（2026-04-06）

- 顶层总览：`Docs/video_pipeline.md`
- 视频 RPM CV 主线细节：`Docs/video_rpm_cv_pipeline.md`
- 当前子项目细节：`Docs/video_manual_annotation_assets.md`
