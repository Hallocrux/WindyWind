# windNotFound

手工标注、21 点 RPM 直接拟合、逐帧 ROI 视频频谱验证子项目。

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
