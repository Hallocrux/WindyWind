# 视频模块设计与验证备注

本文档存放视频 RPM 模块中偏设计、规划和验证策略的内容，不再堆在根目录 `PROJECT.md` / `README.md`。

## 1. RPM 算法解耦重构认知

### 1.1 当前耦合点（已确认）

- 视频主流程入口在 `src/windyWindHowfast/app.py:run_analysis`，其中同时包含：
  - ROI 获取（自动 / 手动 / 历史参数）
  - time-angle 构建
  - 2D 频谱求 RPM
  - 图和 JSON 落盘
- 人工标注流程入口在 `src/windyWindHowfast/annotate.py`，其 RPM 估计逻辑当前耦合在：
  - `src/windyWindHowfast/annotation_summary.py:_estimate_selector_rpm`
- 因此“RPM 估计能力”目前分散在两个上下文里，尚未形成一个统一可复用、可对比、可测试的算法入口。

### 1.2 建议的最小重构目标

- 把“RPM 估计”提升为独立模块（建议新建 `rpm_estimator.py`），只做一件事：
  - 接受标准化输入
  - 输出统一的 RPM 结果结构
- UI/CLI、ROI、落盘都只调用该模块，不在各自文件里重复计算逻辑。
- 先不做“大改目录”，优先做“算法边界解耦 + 接口统一”，避免影响当前可运行链路。

### 1.3 建议接口口径（供后续实现）

- `estimate_rpm_from_time_angle_map(...)`：
  - 输入：`time_angle_map`, `fps`, 频率搜索参数
  - 输出：`RPMEstimate(method="spectral", rpm, signed_rpm, confidence, diagnostics)`
- `estimate_rpm_from_annotations(...)`：
  - 输入：`[(frame_index, center, blade_1), ...]`, `fps`
  - 输出：`RPMEstimate(method="annotation_regression", rpm, signed_rpm, confidence, diagnostics)`
- `compare_rpm_estimates(...)`：
  - 输入：自动算法结果 + 人工标注结果
  - 输出：误差指标（绝对误差、相对误差、状态码）

### 1.4 分阶段实施建议

- Phase 1（低风险）：
  - 把 `_estimate_selector_rpm` 从 `annotation_summary.py` 抽到 `rpm_estimator.py`
  - 保持外部行为不变
- Phase 2（统一入口）：
  - `app.py` 的频谱 RPM 输出也改为调用 `rpm_estimator.py`
  - 两条路径共享 `RPMEstimate` 数据结构
- Phase 3（验证闭环）：
  - 在 `outputs/annotations/<task>/summary.json` 增加 `auto_vs_manual` 对比区
  - 明确哪些片段通过验证、哪些片段失败及原因

### 1.5 验收标准（面向重构）

- 同一段视频可稳定产出两类 RPM：
  - 自动（ROI + 2D 频谱）
  - 人工参考（标注回归）
- 两类结果可程序化比较，不依赖人工读图。
- 任一入口（分析 CLI / 标注 CLI）都不再内嵌 RPM 计算细节，只做数据准备和调用。

## 2. 自动 ROI 失败后的改进方向

### 2.1 已知后续改进方向

- 当前自动 ROI 已经不是单条启发式流程，后续更合理的扩展方向是继续增加 generator，而不是继续堆硬编码规则。
- 当前主峰选择是单峰启发式，后续可考虑：
  - 多半径带融合
  - 时频局部稳健性筛选
  - 转向符号和谐波一致性校验

### 2.2 自动 ROI 第一版当前认知

- 当前默认使用 `sample_reference_frames(...)` 采样参考帧集合，再由多个 generator 并行产出候选。
- 候选统一使用 `ROICandidate(center_x, center_y, radius, score, source, metadata)` 表示。
- 统一评分输出 `ROICandidateScore`，至少包含：
  - `total_score`
  - `center_consistency_score`
  - `radius_consistency_score`
  - `signal_energy_score`
  - `boundary_penalty`
  - `occlusion_penalty`
- 自动 ROI 成功与否不在 generator 内判断，而在统一选择层完成：
  - 选最高分候选
  - 低于阈值则失败
  - 允许交互时回退手动
  - 不允许交互时直接报错
- 当前版本定位是“弱泛化、可扩展地基”，不是只为某个样本写死规则。

### 2.3 自动 ROI 的下一步改进方向排序

- `P1`：新增“轮毂中心”导向的 generator，而不是只找运动圆
  - 可尝试：
    - 长直线/边缘检测后，估计三叶片主方向并求交点
    - 多帧 blade mask 或 skeleton 的交汇点估计
    - 小模型 keypoint / detector，直接回归 hub center
- `P1`：修改参考帧采样策略
  - 不再只取前 `0.62s` 的 `early_stable` 低运动帧
  - 至少要覆盖更长时间，显式追求相位多样性，让 motion map 反映整圈扫掠包络
- `P1`：让 scoring 真正偏向“轮毂”而不是“局部运动团”
  - 增加中心区域的三叶片交汇证据
  - 增加近似 `120°` 间隔的角向结构约束
  - 对“圆心明显落在某一片叶片上”的候选加罚
- `P2`：把下游频谱质量前移到 ROI 选择
  - 对 top-k ROI 先做一小段快速 time-angle 分析
  - 优先保留能产生稳定 `|k|≈3` 及谐波族的候选
  - 不再只看图像空间的局部自洽分数
- `P2`：重做 motion map 的表征
  - 试 `max` 投影、占据图、长时边缘叠加，而不是仅靠相邻帧差分均值
  - 目标是让完整扫掠圆盘比单个叶尖热点更显著
- `P3`：限制 `static_structure` 的搜索区域
  - 例如优先屏蔽底座、地面、人物区域
  - 或加入“离图像底部过近”的惩罚，减少室内背景误检

## 3. 人工标注与自动 RPM 的验证思路

### 3.1 人工标注能否用于验证 RPM 算法可靠性

- 可以用，但当前更适合作为“参考基准”而不是严格意义上的 ground truth。
- 原因不是标注数据不够，而是当前人工标注 RPM 的推导口径本身也带有方法假设：
  - 只使用 `blade_1` 相对 `center` 的角度
  - 默认同一 selector 内角速度近似恒定，用一次线性拟合估计 RPM
  - 还没有使用 `support_a / support_b` 做稳像或相机抖动补偿
- 因此当前人工标注最适合回答的问题是：
  - 自动算法在同一视频片段上，和人工几何参考是否一致
  - 自动算法在不同 selector 上，是否稳定落在合理 RPM 范围内
  - 自动算法失败时，误差主要来自 ROI 偏差、峰值选错，还是视频片段本身不稳定
- 当前人工标注还不太适合直接回答：
  - 绝对 RPM 是否已经达到“工程真值级”精度
  - 算法误差是否已经小到可以替代独立转速传感器
- 现阶段更合理的验证口径是二层：
  - 第 1 层：把人工标注 RPM 当参考值，评估自动算法相对误差
  - 第 2 层：同时检查人工标注自身质量，例如时间跨度、帧数、角度展开是否平滑、残差是否异常
- 推荐的最小评估单元是 selector，而不是整段视频只给一个总 RPM：
  - 因为当前算法和人工拟合都对时间段选择敏感
  - selector 级对齐更容易定位“哪一段失败、为什么失败”
- 当前若要说“算法可靠”，至少应同时满足：
  - 在多个 selector 上与人工参考的误差稳定，而不是只在单个片段碰巧对齐
  - 误差分布没有明显长尾
  - ROI、谱峰模态 `k`、输出 RPM 三者在失败样本上可解释
- 下一步如果要把人工标注从“参考基准”提升到“更接近真值”，优先级应为：
  - 用 `support_a / support_b` 做稳像参考
  - 对人工角度序列输出拟合残差和异常帧诊断
  - 在更长时间跨度上做分段一致性检查，而不是只报单个线性拟合 RPM

### 3.2 `config/test.yaml` 当前可用性复核

- `config/test.yaml` 当前 selector 为：
  - `kind=window`
  - `center=2000`
  - `before=10`
  - `after=10`
  - `step=1`
- 因此实际展开的是 frame `1990-2010`，共 `21` 帧，而不是 `20` 帧。
- 现有实现对这种长度是支持的：
  - `_estimate_selector_rpm(...)` 只要求同一 selector 下至少有 `2` 个不同帧
  - 当前 `outputs/annotations/test/summary.json` 已成功生成 `status=ok`
- 这组现有标注的摘要结果为：
  - `annotated_item_count=21`
  - `frame_span=20`
  - `time_span_sec≈0.336`
  - `rpm≈137.96`
- 但这组窗口时间跨度仍然偏短，因此它更适合做“当前算法在该片段上是否大致一致”的快速核对，不适合单独作为高置信度的可靠性结论。

### 3.3 当前自动 RPM 算法的对外暴露方式

- 当前自动 RPM 主入口是：
  - Python 函数：`src.windyWindHowfast.app.run_analysis(...)`
  - CLI：`python -m src.windyWindHowfast [args...]`
- `run_analysis(...)` 当前返回 `AnalysisResult` dataclass，其中已包含：
  - `rpm`
  - `rotor_freq_hz`
  - `peak_temporal_hz`
  - `peak_spatial_mode`
  - `peak_magnitude`
  - `roi`
  - `roi_source`
  - `roi_detection_status`
- 因此从“功能是否已经暴露”为可调用接口的角度看：
  - 是，已经不是只能走 CLI 文本输出
  - 可以被其他 Python 模块直接 import 并调用
- 但当前 `run_analysis(...)` 仍然是“重型入口”，职责混在一起：
  - 打开整段视频
  - 自动/手动 ROI 解析
  - 全段 `time_angle_map` 构建
  - 结果落盘
  - 可选 UI 展示
- 这意味着它适合命令行分析，不算是最适合嵌入 `annotation_summary.py` 的细粒度 API。

### 3.4 集成到人工标注 `summary.json` 流程的可行性判断

- 可以集成，而且集成点非常明确：
  - `annotate.py` 每次保存后会调用 `write_annotation_summary(...)`
  - `write_annotation_summary(...)` 再调用 `build_annotation_summary_payload(...)`
  - 因此只要在 `annotation_summary.py` 的 selector 汇总阶段补一个“自动算法结果”分支，就能把结果并列写进同一个 `summary.json`
- 当前阻碍不是“做不到”，而是接口颗粒度不合适：
  - 手工标注 summary 是 selector 级、短窗口级流程
  - 自动 RPM 当前默认入口更偏“整段视频分析任务”
- 若要稳妥集成，最合理的重构方向不是在 `annotation_summary.py` 里直接硬调 CLI，而是新增一个更轻的函数，例如：
  - 输入：`video_path`, `start_frame`, `max_frames`, `roi`
  - 输出：`rpm`, `peak_temporal_hz`, `peak_spatial_mode`, `peak_magnitude` 以及失败原因
- 对接人工标注时，ROI 也不该优先依赖自动 ROI：
  - 更合理的是直接用人工标注里的 `center` 估计 selector ROI 中心
  - 半径可先由 `center -> blade_i` 的距离统计得到
  - 这样能把验证问题更集中到“频谱 RPM 算法本身”，而不是再次混入 ROI 误差
- 结论：
  - 工程上“方便集成”，前提是先把自动 RPM 主链拆出一个无副作用、selector 级的轻量 API
  - 如果不拆，直接复用 `run_analysis(...)` 也能接，但会把落盘、副作用和 ROI 选择逻辑一起带进 summary 生成流程，不够干净
