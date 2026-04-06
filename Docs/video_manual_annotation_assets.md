# 视频手工标注资产与验证主线

- 状态：`current`
- 首次确认：`2026-04-06`
- 最近复核：`2026-04-06`
- 代码口径：`src/windNotFound/`
- 数据范围：`data/video/VID_20260330_162635.mp4`
- 证据入口：
  - `config/test.yaml`
  - `outputs/annotations/test/annotations.jsonl`
  - `outputs/annotations/test/summary.json`
  - `outputs/annotations/test/video_rpm_eval.json`

## 1. 模块定位

这个文档只展开 `src/windNotFound/` 的手工标注、RPM 拟合和视频侧验证细节。

当前角色：

- 管理待标注 selector；
- 把逐帧人工点位记录为原始标注数据；
- 从标注点位拟合 selector 级 RPM；
- 用逐帧标注 ROI 反向验证视频频谱主线。

## 2. 关键资产

### 2.1 [2026-04-06] `annotations.jsonl` 是当前工况5视频手工标注的原始事实源

- 状态：`current`
- 首次确认：`2026-04-06`
- 最近复核：`2026-04-06`

当前应明确区分：

- `outputs/annotations/test/annotations.jsonl`
  - 原始逐帧标注记录
  - 是当前最底层、最关键的人工资产
- `outputs/annotations/test/summary.json`
  - 基于 `jsonl` 计算得到的 selector 级 RPM 汇总
- `outputs/annotations/test/video_rpm_eval.json`
  - 基于逐帧标注 ROI 对视频频谱主线做的二次验证结果

如果后续出现“手工标注结果和汇总结论不一致”的问题，应优先回看：

- `annotations.jsonl`

而不是先相信汇总文件。

### 2.2 [2026-04-06] 当前 `test` 任务是工况5视频片段的关键人工资产

- 状态：`current`
- 首次确认：`2026-04-06`
- 最近复核：`2026-04-06`

当前 `test` 任务对应：

- 视频：`data/video/VID_20260330_162635.mp4`
- selector 类型：`window`
- 帧范围：`1975-2025`
- 标注帧数：`51`
- 时间跨度：约 `0.8390s`
- 当前手工拟合 RPM：`147.4341`
- 当前视频 FFT RPM：`140.2224`

这组资产的用途不是“替代 manifest 标签”，而是：

- 作为 `工况5` 视频片段上的高价值参考证据；
- 用来复核 CV 主线是否至少落在合理量级；
- 用来暴露“视频片段局部状态”和“表格全工况标签”之间的可能差异。

## 3. 当前工作流

当前手工标注与验证流程：

1. 用 YAML 定义视频源与 selector；
2. 用标注 UI 逐帧记录点位，写入 `annotations.jsonl`；
3. 基于 `center -> blade_1` 角度序列拟合 RPM，输出 `summary.json`；
4. 对同一 selector 内每帧使用人工 ROI，构造 `time_angle_map`；
5. 输出 `video_rpm_eval.json`，与手工 RPM 对比。

## 4. 当前输入输出

入口命令：

```bash
uv run python src/windNotFound/run_annotate.py --task config/test.yaml
uv run python src/windNotFound/run_fit_rpm.py --task config/test.yaml
uv run python src/windNotFound/run_eval_video_rpm.py --task config/test.yaml --selector-index 0
```

关键输出：

- `outputs/annotations/test/annotations.jsonl`
- `outputs/annotations/test/summary.json`
- `outputs/annotations/test/video_rpm_eval.json`

## 5. 当前对齐与解释限制

### 5.1 [2026-04-06] 手工标注片段是局部片段，不等于工况5全段

- 状态：`current`
- 首次确认：`2026-04-06`
- 最近复核：`2026-04-06`

这意味着：

- 手工 RPM 只代表 `51` 帧片段附近的局部状态；
- 它不应被直接解释成 `工况5` 全段唯一真值。

### 5.2 [2026-04-06] 当前仍需要把 `jsonl` 与表格 `time` 列的正式对齐规则写清楚

- 状态：`current`
- 首次确认：`2026-04-06`
- 最近复核：`2026-04-06`

当前仓库里已经有两端关键信息：

- 视频侧原始帧标注：`annotations.jsonl`
- 表格侧绝对时间：`工况5.csv` 的 `time` 列

但当前文档里还没有一份正式说明，明确写出：

- 视频片段如何映射到表格 `time` 列；
- 哪种假设是临时近似；
- 哪种口径才算硬同步。

这部分应继续补档，否则后续很容易再次误把近似对齐当成正式结论。

## 6. 相关探索链条

- `config/test.yaml`
  - 当前关键人工标注任务入口
- `outputs/annotations/test/`
  - 当前关键人工标注资产目录
- `src/try/022_case5_video_manual_label_validation/`
  - 工况5视频手工标注定向验证
- `Docs/video_pipeline.md`
  - 上层总览文档
- `Docs/video_rpm_cv_pipeline.md`
  - 视频 RPM CV 主线详细文档
