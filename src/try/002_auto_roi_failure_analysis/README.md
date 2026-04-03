# 002 自动 ROI 失败分析

## 目标

- 针对 `outputs/windyWindHowfast/VID_20260330_162635/` 这段样本，分析自动 ROI 为什么会选错区域。
- 区分失败主要发生在：
  - 候选生成阶段没有提出接近轮毂的候选；
  - 还是评分/选择阶段把错误候选排到了前面。
- 把结论固化为可复现脚本，而不是只停留在截图观察。

## 输入

- 自动 ROI 候选与评分：
  - `outputs/windyWindHowfast/VID_20260330_162635/analysis_roi_candidates.json`
  - `outputs/windyWindHowfast/VID_20260330_162635/analysis_roi_detection.json`
- 参考 ROI：
  - 默认复用 `cycle_a_roi.json`
  - 它来自此前人工指定的轮毂附近 ROI，可作为“近似正确 ROI”参考，而不是严格真值
- 原视频：
  - `data/video/VID_20260330_162635.mp4`

## 运行方式

```powershell
uv run python src/try/002_auto_roi_failure_analysis/analyze_auto_roi_failure.py
```

也可以显式指定输入输出：

```powershell
uv run python src/try/002_auto_roi_failure_analysis/analyze_auto_roi_failure.py `
  --sample-dir outputs/windyWindHowfast/VID_20260330_162635 `
  --reference-roi outputs/windyWindHowfast/VID_20260330_162635/cycle_a_roi.json `
  --video data/video/VID_20260330_162635.mp4 `
  --output-dir outputs/try/002_auto_roi_failure_analysis
```

## 输出

- `outputs/try/002_auto_roi_failure_analysis/candidate_summary.csv`
  - 每个候选的分数、中心偏差、半径偏差和与参考 ROI 的 IoU
- `outputs/try/002_auto_roi_failure_analysis/top_candidates_overlay.png`
  - 在首帧上叠加参考 ROI、自动选中 ROI 和前几名候选
- `outputs/try/002_auto_roi_failure_analysis/summary.json`
  - 关键统计量
- `outputs/try/002_auto_roi_failure_analysis/summary.md`
  - 面向仓库维护的简短结论
