# Docs 目录说明

这里存放不适合继续堆在根目录 `PROJECT.md` 和 `README.md` 中的内容。

- `PROJECT.md`：只保留当前仍有效的项目状态、默认入口、稳定结论、工程约定。
- `README.md`：只保留用户现在能直接使用的能力、运行方法、输入输出说明、已确认限制。
- `Docs/`：存放设计讨论、重构方案、路线规划、方法教学拆解、风险分析等详细文档。

当前文档拆分如下：

- `data_catalog.md`
  - 数据目录、manifest 规则、工况资产、已知异常字段
- `table_pipeline.md`
  - 表格主线的数据加载、清洗、切窗、特征、train/eval/inference 规则
- `data_quality_and_findings.md`
  - 带时间戳的稳定事实、历史结论、替代关系
- `video_pipeline.md`
  - 视频 RPM 模块与手工标注子项目的当前状态

- `project_roadmap_and_method_notes.md`
  - 项目阶段拆分
  - 建模建议
  - 风险与预处理策略
  - 第一轮表格实验的方法讲解
- `video_module_design_notes.md`
  - 视频 RPM 模块的重构想法
  - 自动 ROI 失败机理分析后的改进方向
  - 人工标注与自动 RPM 对比验证方案
- `howfast_refactor_final_spec_2026-04-03.md`
  - HowFast 重构后的最终结构与固定决策
  - 三条管线职责边界
  - 输出语义与一次切换迁移说明
