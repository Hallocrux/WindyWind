# windyWindHowfast

视频转速分析子模块，当前采用“极坐标展开 + 时空二维频谱”的方法估计风轮转速。

## 目标

- 保留角度维度上的结构信息，不再把整圈角度平均成单一强度信号。
- 在极坐标空间中，把叶片旋转视为“角度轴上的平移”。
- 通过时间频率 `f` 与角向模态 `k` 的二维谱峰，反推出转子频率 `f / k`。

## 当前 ROI 框架

- 自动 ROI 已改成三层结构：
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

关闭交互回退，只测试自动 ROI：

```bash
uv run python -m src.windyWindHowfast \
  --video data/video/VID_20260330_162635.mp4 \
  --no-interactive \
  --no-show \
  --run-name auto_roi_smoke
```

显式指定 ROI 时，会跳过自动候选框架：

```bash
uv run python -m src.windyWindHowfast \
  --video data/video/VID_20260330_162635.mp4 \
  --center-x 455 \
  --center-y 800 \
  --radius 230
```

## 关键 CLI 参数

- `--auto-roi / --no-auto-roi`
- `--interactive / --no-interactive`
- `--roi-frame-strategy`
- `--roi-reference-max-frames`
- `--roi-score-threshold`
- `--roi-debug / --no-roi-debug`
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
