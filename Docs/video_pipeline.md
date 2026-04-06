# 视频主线说明

本文档记录视频 RPM 模块与手工标注子项目的当前工程状态。  
视频主线和表格主线彼此相关，但当前作为独立模块维护。

## 1. 视频 RPM 模块当前状态

- 状态：`current`
- 首次确认：`2026-04-03`
- 最近复核：`2026-04-05`

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

## 2. ROI 机制与当前限制

- 状态：`current`
- 首次确认：`2026-04-03`
- 最近复核：`2026-04-05`
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
- 最近复核：`2026-04-05`

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

## 5. 相关入口

- 当前项目状态总览：`PROJECT.md`
- 表格主线说明：`Docs/table_pipeline.md`
- 历史设计与重构说明：
  - `Docs/video_module_design_notes.md`
  - `Docs/howfast_refactor_final_spec_2026-04-03.md`
