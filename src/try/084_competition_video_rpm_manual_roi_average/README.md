# 084 竞赛测试视频多片段手工 ROI RPM 平均

## 目标

- 针对竞赛测试视频，按“视频结束时间”与对应 CSV 的时间轴做对齐。
- 从视频与 CSV 的重叠稳定区中，自动挑出多个中段片段。
- 每个片段在中心帧上手工标注一个圆形 ROI。
- 每个片段复用 `src/windyWindHowfast/` 的 `time-angle` 二维频谱算法求 RPM。
- 对多个片段的 RPM 做简单平均，得到最终视频 RPM。

## 当前默认口径（2026-04-09）

- 视频结束时间：默认从视频文件名 `YYYYMMDD_HHMMSS` 解析。
- CSV 清洗口径：对齐 `src/current/data_loading.py`
  - 删除首尾连续缺失段；
  - 中间连续缺失 `<=5` 行保留；
  - 中间连续缺失 `>5` 行按长缺失段切开连续区。
- 片段选择口径：
  - 在“视频时间范围”和“CSV 最长重叠连续段”的交集中取片段；
  - 默认取 `3` 个中段片段；
  - 每个片段默认 `51` 帧；
  - 片段中心在重叠区内均匀分布，并避开两端边界。

## 输出

- 任务与上下文：
  - `outputs/try/084_competition_video_rpm_manual_roi_average/tasks/`
- 片段预览图：
  - `outputs/try/084_competition_video_rpm_manual_roi_average/previews/`
- 聚合结果：
  - `outputs/try/084_competition_video_rpm_manual_roi_average/results/`
- 分段 ROI 标注：
  - `outputs/try/084_competition_video_rpm_manual_roi_average/results/<task_stem>_selector_rois.json`
- 如需更细的逐帧手工标注对照，仍可额外复用 `windNotFound` 默认输出：
  - `outputs/annotations/<task_stem>/annotations.jsonl`
  - `outputs/annotations/<task_stem>/summary.json`
  - `outputs/annotations/<task_stem>/video_rpm_eval.json`

## 运行方式

1. 生成多片段标注任务：

```powershell
uv run python src/try/084_competition_video_rpm_manual_roi_average/run_competition_video_rpm_manual_roi_average.py prepare `
  --video "data/video/test/20260409_134805.mp4" `
  --csv "data/test/竞赛预测风速工况.csv"
```

2. 按生成的 task 对每个片段中心帧手工圈 ROI：

```powershell
uv run python src/try/084_competition_video_rpm_manual_roi_average/run_competition_video_rpm_manual_roi_average.py annotate-roi `
  --task "outputs/try/084_competition_video_rpm_manual_roi_average/tasks/competition_video_20260409_134805_manual_roi_average.yaml"
```

3. 标注完成后汇总各片段 RPM 并求平均：

```powershell
uv run python src/try/084_competition_video_rpm_manual_roi_average/run_competition_video_rpm_manual_roi_average.py aggregate `
  --task "outputs/try/084_competition_video_rpm_manual_roi_average/tasks/competition_video_20260409_134805_manual_roi_average.yaml"
```

## 说明

- 这里的最终 RPM 默认使用每个片段的 `video_fft.rpm` 做等权平均。
- 默认主流程是“每片段一个静态手工 ROI”。
- 如果后续想做更强对照，仍可以再用 `src/windNotFound/run_annotate.py` 做逐帧六点标注；本探索的 `aggregate` 会在没有静态 ROI 标注时自动回退到 `windNotFound` 的逐帧 ROI 结果。
- 如果视频文件名不能可靠表达结束时间，可显式传入 `--video-end-time "2026-04-09 13:48:05"`。
