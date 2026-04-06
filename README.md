# windywind

这个 README 面向项目使用者，只保留当前已经可用的功能、运行方式、输入输出位置和已确认限制。详细设计、路线规划和方法讲解见 `Docs/`。

## 当前已实现的内容

- 表格主线：
  - 读取 `data/final/dataset_manifest.csv` 与 `data/final/datasets/工况*.csv`
  - 默认先删除首尾连续缺失段；中间连续缺失 `<=5` 行时线性插值，`>5` 行时直接删除并分段
  - 只在连续段内部按固定窗口切分时序
  - 提取时域/频域特征
  - 做按工况留一验证的风速回归实验
  - 输出默认清洗口径下的实验结果、数据质量报告和自动派生 inventory
- TinyTCN baseline：
  - 在 `src/Baseline_TinyTCN/` 中提供独立的 TinyTCN baseline 入口
  - 复用当前 `50Hz / 5s / 2.5s` 的切窗口径
  - 按工况留一评估，并输出无标签工况预测
- 视频主线：
  - 在 `src/windyWindHowfast/` 中提供独立的视频转速分析 CLI
  - 当前方法是极坐标展开 + `time-angle` 二维频谱
  - 保留完整视频管线：ROI -> RPM
- 标注主线：
  - 在 `src/windNotFound/` 中提供独立的手工标注与 21 点 RPM 拟合工具
  - 用 YAML 定义待标注帧
  - 以 `jsonl` 追加写入标注结果
  - 基于 `center -> blade_1` 角度直接拟合 selector 级 RPM

## 表格实验怎么运行

主入口是根目录 `main.py`。

```bash
uv run python main.py
```

运行后会在 `outputs/` 下写出实验结果。

当前已确认的表格侧事实：

- 输入主数据由 `data/final/dataset_manifest.csv` 与 `data/final/datasets/工况*.csv` 共同组织
- 当前共有 20 个工况文件
- 其中 19 个工况在 manifest 中带有风速和转速标签
- `工况2.csv` 无标签，不参与监督训练
- CSV 文件名只保留工况编号，风速、转速、显示名和备注统一由 manifest 管理
- 宽表主采样间隔约为 `0.02s`，即 `50Hz`
- 已知 `WSMS00005.*` 为错误加速度数据，清洗时应忽略
- 当前默认清洗会删除首尾连续缺失段；中间连续缺失 `<=5` 行时线性插值，`>5` 行时直接删除并切成连续段

运行后当前会写出：

- `outputs/model_summary.csv`
- `outputs/case_level_predictions.csv`
- `outputs/window_level_predictions.csv`
- `outputs/unlabeled_predictions.csv`
- `outputs/data_quality_summary.csv`
- `outputs/data_quality_missing_columns.csv`
- `outputs/dataset_inventory.csv`

当前已完成的 TinyTCN baseline 运行方式：

```bash
uv run python -m src.Baseline_TinyTCN
```

当前 baseline 输出位置：

- `outputs/Baseline_TinyTCN/model_summary.csv`
- `outputs/Baseline_TinyTCN/case_level_predictions.csv`
- `outputs/Baseline_TinyTCN/unlabeled_predictions.csv`

## 视频转速模块怎么运行

默认读取 `data/video/` 下的第一个 `mp4`：

```bash
uv run python -m src.windyWindHowfast
```

指定视频：

```bash
uv run python -m src.windyWindHowfast --video data/video/VID_20260330_162635.mp4
```

显式指定 ROI：

```bash
uv run python -m src.windyWindHowfast \
  --video data/video/VID_20260330_162635.mp4 \
  --center-x 455 \
  --center-y 800 \
  --radius 230
```

每次运行默认把结果写到：

- `outputs/windyWindHowfast/<video_stem>/`

当前常见工件包括：

- `*_roi_candidates.json`
- `*_roi_detection.json`
- `*_analysis_result.json`
- `*_analysis_summary.png`
- `*_first_frame_with_roi.png`
- `*_roi.json`

当前已确认的限制：

- 自动 ROI 在样例 `VID_20260330_162635.mp4` 上已确认会选错到叶片附近的局部高运动圆，而不是轮毂附近的正确 ROI。
- 当前视频 RPM 数值仍需要结合 ROI debug 与频谱图一起判断，不应把单次输出直接当成稳定真值。

## 手工标注与 21 点 RPM 拟合怎么运行

标注：

```bash
uv run python src/windNotFound/run_annotate.py --task config/test.yaml
```

当前任务 YAML 只定义帧选择，支持的 selector：

- `window`
- `range`
- `explicit`

每个任务自动写到：

- `outputs/annotations/<task_stem>/annotations.jsonl`
- `outputs/annotations/<task_stem>/summary.json`

当前固定标注顺序：

1. `support_a`
2. `support_b`
3. `center`
4. `blade_1`
5. `blade_2`
6. `blade_3`

其中：

- `blade_1` 必须是带 marker 的扇叶
- `blade_2`、`blade_3` 需要按顺时针

当前已确认的标注侧事实：

- 同一 `task_item_id` 重做时采用 append-only，读取时以最后一条记录为准
- `summary.json` 当前直接输出 selector 级 RPM 拟合结果
- 标注窗口提示文案当前统一使用英文

基于已有标注直接拟合 RPM：

```bash
uv run python src/windNotFound/run_fit_rpm.py --task config/test.yaml
```

用逐帧标注 ROI 验证视频 RPM 算法：

```bash
uv run python src/windNotFound/run_eval_video_rpm.py --task config/test.yaml --selector-index 0
```

## 已完成探索入口

- `src/try/001_fft_frequency_plot/`
  - 输出 `outputs/try/001_fft_frequency_plot/`
- `src/try/002_auto_roi_failure_analysis/`
  - 输出 `outputs/try/002_auto_roi_failure_analysis/`
- `src/try/013_phase3_cnn_tcn_smoke/`
  - 输出 `outputs/try/013_phase3_cnn_tcn_smoke/`
- `src/try/015_patchtst_loco/`
  - 输出 `outputs/try/015_patchtst_loco/`
- `src/try/016_micn_loco/`
  - 输出 `outputs/try/016_micn_loco/`
- `src/try/017_samformer_loco/`
  - 输出 `outputs/try/017_samformer_loco/`
- `src/try/018_structural_fundamental_frequency_scan/`
  - 输出 `outputs/try/018_structural_fundamental_frequency_scan/`
- `src/try/025_tinytcn_boundary_error_check/`
  - 输出 `outputs/try/025_tinytcn_boundary_error_check/`
- `src/try/026_tinytcn_priority1_quickcheck/`
  - 输出 `outputs/try/026_tinytcn_priority1_quickcheck/`

如果你需要看路线规划、重构想法、方法拆解或详细风险分析，直接阅读 `Docs/`。
