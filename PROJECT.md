# windywind 项目笔记

本文档只面向 coding agent，记录工程现状、数据事实、实验结论和开发约定。
面向低 ML 经验读者的解释性内容已移动到 `README.md`。

## 1. 项目定位

- 这是一个探索性的课程项目，背景材料在 `data/final/00-工程智能基础-工程互联网小项目备课 - 风电5页.pdf`。
- 课程任务的主线有两条：
  - 任务 1：对每条时程数据做基础统计分析，例如最大值、平均值、标准差。
  - 任务 2：基于振动时程数据（加速度、应变）以及风轮转速，预测入流风速。
  - 任务 3：基于振动时程数据识别结构基频；振型、阻尼比属于开放性加分任务。
- 竞赛环节最终会比较：
  - 风速预测值 vs 热线风速仪真实值
  - 基频预测值 vs 真实值

## 2. 当前仓库现状

- 当前仓库几乎还没有形成正式代码工程。
- `main.py` 现在作为项目统一入口，默认调用 `src/current/pipeline.py`。
- `test.ipynb` 处于非常初始的探索状态，只做了按传感器分组和简单画图尝试。
- `README.md` 用于面向项目使用者解释实验方法和背景知识。
- 已新增视频转速分析子模块 `src/windyWindHowfast/`，用于从风机视频中估计转速。
- 顶层 `src/` 下保留了早期模块文件，同时已建立版本分层目录：
  - `src/baseline/`：冻结的 baseline 版本
  - `src/current/`：当前继续开发的版本

## 2.1 当前代码版本约定

- 从 2026-04-02 起，当前可运行版本已冻结到 `src/baseline/`。
- 后续迭代默认只在 `src/current/` 中进行。
- `main.py` 默认指向 `src/current.pipeline.main`，因此：
  - 跑当前开发版本时，直接执行 `python main.py` 或对应的 `uv run python main.py`
  - 若需要复现实验基线，应显式调用 `src/baseline/pipeline.py`
- 这样做的目的，是把“可复现实验基线”和“继续试验的新版本”分离，避免代码口径漂移。

## 2.2 探索目录约定

- 从 2026-04-02 起，`src/try/` 专门用于“一次性探索”工作。
- 每次新的探索都必须在 `src/try/` 下新建独立目录，不和 `src/current/`、`src/baseline/` 混放。
- 推荐命名为“顺序编号 + 主题”，例如：
  - `src/try/001_fft_frequency_plot/`
- 每个探索目录至少要有：
  - 目标与运行说明
  - 一个可直接运行的脚本或 notebook
  - 对应产物写到 `outputs/` 下的同名目录
- 探索阶段应优先复用正式数据加载与清洗逻辑，避免因为探索脚本和主流程口径不一致而误判结果。

## 3. 数据组织

### 3.1 数据目录

- `data/test_data_with_explanation/`
  - 存放较原始的长表数据。
  - 每行是一条单传感器采样记录。
  - 配套有说明文本，部分文件带有风速和转速描述。
- `data/final/datasets/`
  - 存放整理后的宽表数据。
  - 每行是同一时刻的多传感器拼接结果。
  - 更适合作为后续特征工程和建模输入。
- `data/video/`
  - 存放视频法估计风机转速的输入视频。
  - 2026-04-02 已迁移 `VID_20260330_162635.mp4` 作为当前默认样例视频。

### 3.2 原始长表格式

- 字段为：`code`, `type`, `time`, `value1`, `value2`, `value3`
- `type=vibr` 表示加速度传感器：
  - `value1 -> AccX`
  - `value2 -> AccY`
  - `value3 -> AccZ`
- `type=sgd` 表示应变传感器：
  - `value1 -> chdata`
  - `value2`, `value3` 通常为空

### 3.3 整理后宽表格式

- 宽表主键列为 `time`
- 其余列为传感器通道，例如：
  - `WSMS00001.AccX`
  - `WSMS00001.AccY`
  - `WSMS00001.AccZ`
  - `应变传感器1.chdata`
- 大多数工况宽表有 21 列：
  - 1 列时间
  - 5 个加速度计 x 3 轴 = 15 列
  - 5 个应变计 x 1 通道 = 5 列
- `工况5（第三组）.csv` 有 24 列，因为额外出现了 `WSMS00005.AccX/Y/Z`

## 4. 已确认的数据事实

### 4.1 采样频率与时长

- 宽表时间间隔主频为 `0.02s`，即约 `50Hz` 采样。
- `data/final/datasets/` 目前共有 14 个工况文件，其中：
  - 13 个文件名中直接包含风速和转速标签
  - `工况2.csv` 暂无标签
- 当前已确认的宽表长度约为：
  - 工况1：3352 行，约 66.02 秒
  - 工况2：3752 行，约 74.02 秒
  - 工况3：3850 行，约 75.98 秒
  - 工况4：7702 行，约 153.02 秒
  - 工况5：5900 行，约 117.94 秒
  - 工况6：5550 行，约 109.98 秒
  - 工况7：4352 行，约 86.02 秒
  - 工况8：4598 行，约 91.94 秒
  - 工况9：5784 行，约 115.66 秒
  - 工况10：6750 行，约 134.98 秒
  - 工况11：4148 行，约 82.94 秒
  - 工况12：5682 行，约 113.62 秒
  - 工况13：8298 行，约 165.94 秒
  - 工况14：4898 行，约 97.94 秒

### 4.2 缺失值模式

- 所有宽表都存在一定比例缺失值，通常在 `1% - 3%` 左右。
- 缺失更像是多传感器拼接时的不同步或漏采，而不是某一整列完全缺失。
- 2026-04-02 复核 `data/final/datasets/` 后，宽表时间轴比此前判断更干净：
  - 未发现时间解析失败
  - 未发现重复时间戳
  - 未发现排序后非正时间间隔
  - 采样间隔主频稳定为 `0.02s`
- 当前宽表的主要问题不是时间轴跳变，而是缺失值呈“成块出现”，且多发生在序列开头或结尾。
- 现有宽表仍然建议保留排序、去重和间隔检测逻辑，但优先级已经低于“缺失段处理策略”。

### 4.3 已知错误数据

- 已知所有 `WSMS00005`（加速度传感器5）数据如果存在，都是错误值。
- 正确的“加速度传感器5”数据实际在 `WSMS00006`。
- 在 `工况5（第三组）.csv` 中：
  - `WSMS00005.AccX/Y/Z` 大量为全零伪数据
  - `WSMS00006.AccX/Y/Z` 才是有效信号
- 因此后续数据清洗时，必须显式忽略或删除 `WSMS00005.*`，并把 `WSMS00006` 视为有效的第 5 个加速度传感器。

## 5. 标签与监督信号现状

- 当前仓库虽然仍然没有独立的结构化标签表，但 `data/final/datasets/` 的文件名已经携带了大部分任务 2 标签。
- 已可从文件名直接解析出 13 个工况的：
  - `case_id`
  - `wind_speed`
  - `rpm`
- 目前标签范围约为：
  - 风速：`2.12 - 5.38 m/s`
  - 转速：`82 - 230 rpm`
- `工况2.csv` 仍然没有风速与转速标签，因此：
  - 不参与监督训练与交叉验证
  - 可作为训练完成后的无标签推理对象
- 任务 3 所需的结构化真实基频标签目前仍未看到。

## 6. 对项目方向的判断

- 当前最合理的开发顺序不是直接上模型，而是先完成“数据治理 + 可复用分析基线”。
- 现阶段最值得优先做的成果：
  - 原始长表到规范宽表的可复现转换
  - 统一的数据清洗规则
  - 每工况的统计特征与频域特征提取
  - 基频估计的物理启发式基线
  - 一旦补齐标签，即可继续做风速回归模型
- 除了表格时序主线之外，视频法现在可以作为“转速观测”方向的独立支线继续开发：
  - 它不应混入 `src/current/` 现有表格建模 pipeline
  - 应作为独立正式模块维护在 `src/windyWindHowfast/`
  - 后续如果验证稳定，可再考虑和主线任务做特征级或标签级联动

## 11. 2026-04-02 视频转速分析模块迁移

### 11.1 迁移位置

- 代码迁移到：`src/windyWindHowfast/`
- 视频数据迁移到：`data/video/`
- 当前默认样例视频：
  - `data/video/VID_20260330_162635.mp4`

### 11.2 当前算法口径

- 当前较高可用版本不再把“空间结构移动”压成“全局强度变化”。
- 每帧先做风机 ROI 的极坐标展开，把旋转问题转成角度轴上的平移问题。
- 在极坐标图中：
  - 只沿半径维压缩
  - 保留角度维，得到 `angle_profile(theta)`
- 把所有帧堆叠成 `time_angle_map(t, theta)` 后，直接做二维 FFT。
- 频谱搜索时显式排除 `k=0` 的空间均匀项，以抑制整体亮度闪烁或曝光变化。
- 转子频率由主导谱峰 `(f, k)` 反推：
  - `rotor_freq = |f / k|`

### 11.3 当前工程状态

- 该模块当前已从“只能交互演示”推进到“可做测试轮回”的 CLI 版本。
- `src/windyWindHowfast/` 当前已按职责拆分为多个模块，而不是继续堆在单文件中：
  - `constants.py`：默认参数与路径常量
  - `models.py`：ROI / 分析结果 dataclass
  - `support.py`：通用工具函数
  - `analysis_core.py`：极坐标展开、2D 频谱与结果图
  - `roi_detection.py`：候选生成、评分、选择与回退
  - `outputs.py`：debug 与分析结果落盘
  - `app.py`：`run_analysis` 与 CLI 入口
  - `frequency_analysis.py`：兼容层，对外 re-export
- ROI 获取阶段已重构成可扩展框架，分为三层：
  - candidate generators
  - candidate scoring
  - selection + fallback
- 当前已实现的 generator：
  - `motion`
  - `static_structure`
- 已预留空接口：
  - `detector`
- 现有显式 ROI 与历史 `roi.json` 仍然保留，优先级高于自动 ROI。
- 当前关键 CLI 参数包括：
  - `--center-x --center-y --radius`
  - `--roi-json`
  - `--auto-roi / --no-auto-roi`
  - `--interactive / --no-interactive`
  - `--roi-frame-strategy`
  - `--roi-reference-max-frames`
  - `--roi-score-threshold`
  - `--roi-debug / --no-roi-debug`
  - `--run-name`
  - `--start-frame`
  - `--max-frames`
  - `--inner-radius-ratio`
  - `--no-show`
- 当前默认命令：
  - `python -m src.windyWindHowfast`
- 也可显式指定视频：
  - `python -m src.windyWindHowfast --video data/video/VID_20260330_162635.mp4`
- 每次运行默认会把以下工件写到 `outputs/windyWindHowfast/<video_stem>/`：
  - `<run-name>_roi_candidates.json`
  - `<run-name>_roi_detection.json`
  - `<run-name>_roi_best_debug.png`
  - `<run-name>_roi_candidate_<idx>.png`
  - `<run-name>_analysis_result.json`
  - `<run-name>_analysis_summary.png`
  - `<run-name>_first_frame_with_roi.png`
  - `<run-name>_roi.json`

### 11.4 已知后续改进方向

- 当前自动 ROI 已经不是单条启发式流程，后续更合理的扩展方向是继续增加 generator，而不是继续堆硬编码规则。
- 当前主峰选择是单峰启发式，后续可考虑：
  - 多半径带融合
  - 时频局部稳健性筛选
  - 转向符号和谐波一致性校验

### 11.4.1 自动 ROI 第一版当前认知

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

### 11.4.2 2026-04-03 对 `VID_20260330_162635` 的自动 ROI 失败新认知

- 已通过 `src/try/002_auto_roi_failure_analysis/` 对 `analysis` 这轮自动 ROI 做了量化复核。
- 当前使用的人工参考 ROI 为：
  - `center_x=455`
  - `center_y=800`
  - `radius=230`
- 自动 ROI 实际选中的候选为：
  - `center_x=745`
  - `center_y=705`
  - `radius=271`
  - `score=0.5208`
- 新确认的事实：
  - 自动选中圆心距人工参考轮毂约 `305.4 px`
  - 与参考 ROI 的圆形 IoU 只有 `0.157`
  - 全部 `20` 个候选中：
    - 落在参考中心 `120 px` 内的候选数为 `0`
    - IoU `>= 0.5` 的候选数为 `0`
    - 与参考最接近的候选是 rank `#8`，中心误差仍约 `127.3 px`
- 这说明本样本上的自动 ROI 失败不是单纯“阈值太低”：
  - 即使把阈值提到 `0.95`，也只是让这轮选择从 `auto_selected` 变成 `failed`
  - 但 best candidate 仍然是同一个偏到右上叶片附近的错误候选
- 失败机制现在可以拆成两层：
  - 候选生成先失败：现有 generator 几乎没有提出真正靠近轮毂的候选
  - 评分选择再放大偏差：错误候选因为局部运动环稳定，拿到了很高的 `center_consistency_score` 和 `radius_consistency_score`
- 根因判断：
  - `motion` generator 当前是对运动 blob 做 `minEnclosingCircle`，天然更容易围住“叶尖/叶片扫过的局部高运动区”，不保证圆心落在轮毂
  - `support_refined_radius` 只会细化半径，不会纠正中心，因此一旦初始中心偏到叶片，后续仍会沿着错误中心自洽
  - 当前 `center_consistency_score` 的本质是在检测“候选中心附近的局部质心是否稳定”，这会把叶片局部运动团也当成高质量中心
  - 当前评分只奖励“稳定的圆环信号”，没有编码“三叶片应围绕共同轮毂旋转”的几何约束
  - `static_structure` generator 在这个室内场景里也会产出很多背景伪候选，例如底座、墙角、人物附近结构，噪声较大
  - `early_stable` 参考帧当前只落在 frame `0-37`，对应约前 `0.62s`，时间覆盖过短，motion map 更像是短时局部运动，而不是完整叶轮扫掠包络
- 对后续改进优先级的判断：
  - 第一优先级不是继续调阈值，而是补“更接近轮毂定义”的 generator
  - 第二优先级是在 scoring 中补几何先验和下游频谱一致性约束
  - 第三优先级才是继续微调现有 motion/static_structure 的权重

### 11.5 2026-04-02 CLI 测试轮回结果

- 已使用 `data/video/VID_20260330_162635.mp4` 跑通一轮“先定 ROI，再复用 ROI 调参”的测试工作流。
- 本轮固定 ROI 为：
  - `center_x=455`
  - `center_y=800`
  - `radius=230`
- 测试工件位于：
  - `outputs/windyWindHowfast/VID_20260330_162635/`
- 两轮代表性测试结果如下：
  - `cycle_a`
    - 参数：`start_frame=0`, `max_frames=300`, `inner_radius_ratio=0.55`
    - 输出：`2.98 RPM`
    - 谱峰：`peak_temporal_hz=0.3973`, `k=8`
  - `cycle_b`
    - 参数：`start_frame=500`, `max_frames=1200`, `inner_radius_ratio=0.55`, `min_temporal_hz=1.0`
    - 输出：`435.04 RPM`
    - 谱峰：`peak_temporal_hz=7.2507`, `k=-1`
- 当前新增认知：
  - CLI 工作流已经能稳定支撑“显式 ROI -> 落盘 -> 复用 ROI -> 分段调参”。
  - 但算法结果仍明显不稳定，且对时间段选择和谱峰搜索下界很敏感。
  - 从结果图看，时空图里既有低频慢漂移，也有疑似叶片通过频率或谐波主导项；当前直接用单个主峰做 `f / k` 很容易选错物理分量。
  - `cycle_a_first_frame_with_roi.png` 显示当前 ROI 圆心偏左，圆内混入了轮毂附近结构和背景墙边缘，这会直接破坏“旋转 -> 角度平移”的极坐标前提。
  - 当前风机是三叶片结构，后续峰值筛选不应允许 `k=1` 或其他低阶伪模态无约束地主导最终转频；至少应把 `|k|≈3` 及其谐波作为更高优先级候选。
  - 下一步不该继续只靠肉眼换几个参数，而应优先补：
    - 多半径带融合
    - 谐波族一致性校验
    - 分段稳定性评分
    - 更明确地区分“转频”与“叶片通过频率”

### 11.6 自动 ROI 的下一步改进方向排序

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

## 7. 推荐的近期工作拆分

### 7.1 第一阶段：数据治理

- 建立统一的数据加载器，兼容原始长表和最终宽表。
- 标准化时间列格式，转成 `datetime64[ns]` 或相对秒。
- 排序、去重、检测异常时间间隔。
- 统一传感器命名。
- 显式处理 `WSMS00005` 错误数据问题。
- 输出干净、可复现的分析输入表。

### 7.2 第二阶段：探索分析

- 计算每工况、每传感器通道的：
  - 最大值
  - 平均值
  - 标准差
  - RMS
  - 峰峰值
- 绘制时域曲线、频谱、PSD。
- 对比不同工况下主频峰值和能量分布。

### 7.2.1 当前已启动的第一个探索

- 已新增目录：`src/try/001_fft_frequency_plot/`
- 目标：先把“时域信号 FFT 后的频域图像”稳定跑通。
- 当前默认参数：
  - 工况：`工况1`
  - 通道：`WSMS00001.AccX`
  - 输出：`outputs/try/001_fft_frequency_plot/case1_WSMS00001_AccX_fft.png`
- 当前口径：
  - 直接复用 `src/current/data_loading.py` 的清洗逻辑
  - 对整段时域信号减均值后做 `numpy.fft.rfft`
  - 先输出单边幅值谱，后续如有需要再补 PSD、加窗和多工况对比

### 7.2.2 自动 ROI 失败诊断探索

- 已新增目录：`src/try/002_auto_roi_failure_analysis/`
- 目标：对 `VID_20260330_162635` 这段样本量化分析“自动 ROI 为什么会选错圆”。
- 当前输入口径：
  - 自动候选：`analysis_roi_candidates.json`
  - 自动检测结果：`analysis_roi_detection.json`
  - 参考 ROI：复用此前人工指定的 `cycle_a_roi.json`
- 当前默认输出到：
  - `outputs/try/002_auto_roi_failure_analysis/`
- 当前产物包括：
  - `candidate_summary.csv`
  - `top_candidates_overlay.png`
  - `summary.json`
  - `summary.md`

### 7.3 第三阶段：任务基线

- 风速预测基线：
  - 在标签不足前，先只搭建特征提取和训练接口。
  - 等标签补齐后，可先试线性回归、随机森林、梯度提升树。
- 基频识别基线：
  - 基于 FFT / PSD 主峰估计基频。
  - 再考虑更稳健的峰值筛选和多传感器融合。

### 7.4 第四阶段：交付物

- 清晰的数据处理脚本
- 可复现实验 notebook 或命令行入口
- 每个工况的分析报告图表
- 模型训练与评估脚本
- 用于汇报的结果汇总表

## 8. 当前关键风险

- 任务 2 已经不再是“完全无标签”，但样本本质上仍是小样本工况级学习。
- 若把同一工况切出来的多个窗口随机打散做训练/验证，会发生严重信息泄漏；必须按工况划分验证集。
- 转速对风速预测的解释力很强，容易让模型退化成“主要靠转速做映射”，从而掩盖振动信号本身的信息量。
- 采样时序存在局部异常，若不先清洗，会污染频域特征和模型输入。
- 传感器命名与真实物理位置映射目前仍不完整，需要后续补充。

## 9. 任务 2 的模型选择建议

- 任务 2 当前本质上是“小样本回归”问题，而不是深度学习问题。
- 在目前仓库信息下，更合理的优先级是：
  - 第一选择：`LightGBM` 或 `XGBoost` 这类基于表格特征的梯度提升树回归模型
  - 第二选择：`RandomForestRegressor`
  - 必做基线：`LinearRegression` / `Ridge`
- 不建议一开始就用：
  - `LSTM`、`Transformer`、`1D-CNN`
  - 原因是正式工况数量太少，极易过拟合，且解释性和调试成本都偏高
- 更稳妥的建模方式应是：
  - 先把每个工况或每个时间窗转成统计特征、频域特征、转速特征
  - 再用表格回归模型预测风速
- 若后续标签扩展到大量时间窗样本，推荐优先尝试：
  - `LightGBMRegressor`
  - `XGBoostRegressor`
  - `CatBoostRegressor`
- 若最终标签仍然只有“每工况一个风速”，则要非常谨慎：
  - 深度模型基本不合适
  - 线性模型、岭回归、树模型配合严格交叉验证会更可信

## 10. 2026-04-02 第一轮对比实验结果

- 已实现任务 2 的第一轮可复现实验链路：
  - `src/data_loading.py`：扫描、标签解析、清洗
  - `src/features.py`：按 `5s` 窗长、`2.5s` 步长切窗并提取时域/频域特征
  - `src/experiment.py`：按工况留一验证、模型对比、结果汇总
  - `main.py`：单一入口，执行全流程并写出 `outputs/*.csv`
- 本轮共生成 577 个窗口样本，评估协议为 `Leave-One-Condition-Out`。
- 模型结果如下：
  - `LinearRegression + RPM_ONLY`：`case_mae = 0.2530`
  - `Ridge + VIB_FT_RPM`：`case_mae = 0.3327`
  - `Ridge + VIB_FT`：`case_mae = 0.4399`
  - `HistGradientBoostingRegressor + VIB_FT`：`case_mae = 0.4552`
  - `RandomForestRegressor + VIB_FT`：`case_mae = 0.4668`
  - `HistGradientBoostingRegressor + VIB_FT_RPM`：`case_mae = 0.5776`
  - `RandomForestRegressor + VIB_FT_RPM`：`case_mae = 0.6060`
- 当前结论：
  - 若任务定义允许直接使用转速，则第一轮最佳模型实际上是 `rpm-only LinearRegression`
  - 若目标是评估“仅靠振动信号能否预测风速”，当前最强的 rpm-free 方案是 `HistGradientBoostingRegressor + VIB_FT`
  - `工况2.csv` 缺少 rpm，因此最终对它的无标签推理采用了最佳 rpm-free 模型，而不是全局最优模型
- 当前对 `工况2.csv` 的预测结果为：`3.9253 m/s`

## 12. 2026-04-02 数据质量复核

### 12.1 本轮复核范围

- 已新增 `src/data_quality.py`，用于对 `data/final/datasets/` 的 14 个工况做可复现质量检查。
- 结果已写出到：
  - `outputs/data_quality_summary.csv`
  - `outputs/data_quality_missing_columns.csv`

### 12.2 当前最重要的数据事实

- 14 个工况在“共有 20 个有效信号通道”上的平均缺失率为 `1.6113%`。
- 缺失率最高的是工况 3，为 `2.4026%`。
- 最长连续缺失段为 `102` 点，约 `2.04s`。
- 当前 `5s` 切窗下，受原始缺失影响的窗口占比约 `3.99%`。
- 各工况通常只会有 `1 - 2` 个窗口受缺失影响，但这些窗口的最差缺失比例可达 `11.36% - 25.36%`。
- 这些缺失窗口主要集中在序列开头和结尾，因此：
  - 对整体样本量影响不大
  - 但对窗口级 FFT 峰值、频带能量、RMS 等特征会有明显污染风险

### 12.3 对现有预处理是否足够的判断

- 如果目标只是“先跑通第一轮基线实验”，当前预处理是够用的：
  - 已完成时间清洗、排序、去重
  - 已完成统一数值化和缺失补齐
  - 可以稳定产出无 NaN/Inf 的特征矩阵
- 如果目标是“让频域特征更可信、让实验结果更可解释”，当前预处理还不够：
  - 目前对所有缺失一律做线性插值 + `ffill/bfill`
  - 这会把开头或结尾长达 `1 - 2s` 的缺失段直接补成平滑段或常值段
  - 对统计量影响有限，但会扭曲 FFT 主峰、频带能量占比和部分振动幅值特征

### 12.4 下一步更合理的预处理策略

- 保留现有 `WSMS00005.*` 剔除规则，不再把它们纳入有效输入。
- 新增“缺失段分级处理”而不是统一插值：
  - 短缺口可插值
  - 长缺口，尤其是首尾缺口，优先裁剪或直接丢弃受影响窗口
- 在切窗前增加质量标记：
  - 每窗口缺失比例
  - 是否接触首尾缺失段
  - 每通道最大连续缺失长度
- 训练时默认过滤掉受重度缺失影响的窗口，再和“不过滤版本”做对比实验。
- 若后续要重点做频域分析或基频识别，建议优先采用“裁边 + 丢弃脏窗口”，而不是简单全量插值。

### 12.5 已实现的“过滤脏窗口”对比实验

- 目前代码已在切窗阶段输出窗口质量标签：
  - `raw_missing_ratio`
  - `raw_missing_rows`
  - `touches_leading_missing`
  - `touches_trailing_missing`
- `main.py` 现在会额外跑一版过滤实验，结果写到 `outputs/filtered/`。
- 当前过滤规则是直接丢弃：
  - `raw_missing_ratio > 0` 的窗口
  - 或触碰首尾缺失段的窗口
- 过滤后各带标签工况通常只减少 `1 - 2` 个窗口，删除比例约 `1.67% - 6.25%`。
- 过滤前后模型结论对比如下：
  - `RPM_ONLY + LinearRegression` 仍然是全局最优，`case_mae` 基本不变：`0.2530 -> 0.2531`
  - `Ridge + VIB_FT_RPM` 明显改善：`0.3327 -> 0.3115`
  - `Ridge + VIB_FT` 改善：`0.4399 -> 0.4158`
  - `HistGradientBoostingRegressor + VIB_FT` 只小幅改善：`0.4552 -> 0.4515`
- 这说明：
  - 缺失污染窗口数量不多
  - 但足以影响 rpm-free 模型排序和无标签工况推理结果
- 过滤后，最佳 rpm-free 模型已从 `HistGradientBoostingRegressor + VIB_FT` 变为 `Ridge + VIB_FT`。
- 对 `工况2.csv` 的无标签预测也从 `3.9253 m/s` 变为 `3.0078 m/s`，表明当前无标签推理对预处理口径较敏感。
