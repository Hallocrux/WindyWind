# WindyWind HowFast 重构最终 Spec（2026-04-03）

## 1. 设计目标

- 提升 ROI / RPM 算法可复用性与可扩展性。
- 维持管线编排简单，不引入重型 Service 架构。
- 以视频为统一核心资产，明确标注流程与分析流程边界。

## 2. 固化决策

- 仅抽象稳定变化点：`ROIAlgorithm`、`RPMAlgorithm`。
- 强制执行顺序：先 ROI，再 RPM。
- RPM 算法不得自行决定 ROI 策略。
- 标注管线与 RPM 分析管线分离，默认 `summary.json` 不含 RPM。
- 如需“基于标注结果算 RPM”，必须走独立二次分析管线。

## 3. 代码结构

- `src/windyWindHowfast/domain/`
  - `models.py`: `VideoAsset`、`ROIResult`、`RPMResult`、`AnnotationRecord`、`AnnotationSummary`
- `src/windyWindHowfast/roi/`
  - `base.py`: `ROIAlgorithm`
  - `manual.py`: `ManualROIAlgorithm`
  - `auto.py`: `AutoROIAlgorithm`
  - `annotation_center.py`: `AnnotationCenterROIAlgorithm`
- `src/windyWindHowfast/rpm/`
  - `base.py`: `RPMAlgorithm`
  - `spectral_fft.py`: 当前 RPM 实现
- `src/windyWindHowfast/pipelines/`
  - `rpm_pipeline.py`
  - `annotation_pipeline.py`
  - `annotation_analysis_pipeline.py`
- `src/windyWindHowfast/io/`
  - `video_loader.py`
  - `annotation_writer.py`
  - `summary_writer.py`
- `src/windyWindHowfast/cli/`
  - `main.py`（命令分发）
  - `analyze.py`
  - `annotate.py`
  - `annotation_analysis.py`

## 4. 文件语义

- `outputs/annotations/<task_stem>/annotations.jsonl`：逐条标注记录（append-only）。
- `outputs/annotations/<task_stem>/summary.json`：仅标注汇总，不含 RPM。
- `outputs/annotations/<task_stem>/analysis.json`：标注后二次分析结果。
- `outputs/windyWindHowfast/<video_stem>/...`：RPM 管线分析工件与 debug 结果。

## 5. 迁移策略

- 采用一次切换：新架构与新语义直接生效。
- `python -m src.windyWindHowfast` 默认进入 RPM 分析管线。
- `python -m src.windyWindHowfast annotate` 进入标注管线。
- `python -m src.windyWindHowfast annotation-analysis` 进入标注后二次分析管线。
