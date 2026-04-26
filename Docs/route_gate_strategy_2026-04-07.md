# Route / Gate 路线选择策略（2026-04-07）

- 状态：`current`
- 首次确认：`2026-04-07`
- 最近复核：`2026-04-07`
- 数据范围：
  - 主训练域：`data/final/` 的带标签工况
  - 外部验证域：`data/added/` 的 `工况21-24`
- 关联代码口径：
  - `src/try/041_rpm_vs_learned_midband_check/`
  - `src/try/042_rpm_learned_midband_multiseed_stability_check/`
  - `src/try/043_2_fft_rpm_to_wind_replay/`
  - `src/try/043_3_fft_midband_fusion_replay/`
  - `src/try/044_final_loco_fft_midband_fusion_check/`
- 证据入口：
  - `Docs/experiments/fft_midband_fusion_replay_2026-04-07.md`
  - `outputs/try/043_3_fft_midband_fusion_replay/stability_overview.csv`
  - `outputs/try/044_final_loco_fft_midband_fusion_check/seed_runs/`

## 1. 目标

本文档回答 `2026-04-07` 这条路线问题：

- 如果 `midband(3.0-6.0Hz)` 在 `added` 上有价值、但在 `final` 上可能拖后腿，后续该如何组织一条更通用的部署路线；
- 是否应该直接做“`final` / `added` 二分类”；
- 除了 `TCN` 之外，是否存在更合适的模型来自动学习这个判断标准。

## 2. 当前问题的更准确表述

### 2.1 [2026-04-07] 当前真正需要学习的不是“域标签”，而是“哪条路线更适合当前样本”

- 状态：`current`
- 首次确认：`2026-04-07`
- 最近复核：`2026-04-07`

截至 `2026-04-07`：

- `043_3` 已证明：
  - `FFT + learned midband` 在 `added` 上能带来增益；
- `044` 当前已完成的前 `3` 个 seed 显示：
  - `final LOCO` 上，`FFT + learned midband` 暂未优于 FFT 解析单支线。

这说明更合理的问题不是：

- “这个样本是不是 `final` 域”
- “这个样本是不是 `added` 域”

而是：

- “这个样本更应该走 `base route`，还是 `enhanced route`”

更具体地说：

- `base route`
  - 更偏向 `final` 默认稳健路线
- `enhanced route`
  - 更偏向 added 上收益更强的增强路线

因此当前更推荐的目标是学习：

- `route selection`
- 或 `fusion gate`

而不是直接训练一个“`final/added` 域分类器”。

## 3. 推荐的建模形式

### 3.1 [2026-04-07] 优先做 route / gate，而不是直接做硬域分类

- 状态：`current`
- 首次确认：`2026-04-07`
- 最近复核：`2026-04-07`

当前更推荐的两种形式是：

#### 形式 A：硬路由

- 输出：
  - `route = base`
  - 或 `route = enhanced`

部署时：

- 若 `route = base`
  - 走 `final` 默认路线
- 若 `route = enhanced`
  - 走 `added` 增强路线

#### 形式 B：软门控

- 输出：
  - `w in [0, 1]`

最终预测写成：

- `pred = (1 - w) * pred_base + w * pred_enhanced`

截至 `2026-04-07`，更推荐先做：

- 软门控

原因是：

- 当前 `midband` 的价值不是“永远开”或“永远关”；
- 更像在部分样本 / 部分工况机制下才值得加大权重；
- 软门控比硬分类更容易容纳“处在两域之间”的样本。

### 3.2 [2026-04-07] 当前更推荐先做 case 级 gate，再决定是否下沉到 window 级

- 状态：`current`
- 首次确认：`2026-04-07`
- 最近复核：`2026-04-07`

当前数据规模仍偏小，因此更推荐的顺序是：

1. 先做 `case-level gate`
2. 若 case 级规则成立，再考虑 `window-level gate`

不建议直接一开始就做：

- 端到端的 window 级门控网络

原因是：

- 当前工况数有限；
- window 级训练样本虽然更多，但标签本质上仍高度相关于工况级结构；
- 过早做端到端门控，容易把问题重新变成高方差的深度学习拟合。

## 4. gate 的监督信号应该怎么定义

### 4.1 [2026-04-07] 不建议把“数据目录来源”直接当标签

- 状态：`current`
- 首次确认：`2026-04-07`
- 最近复核：`2026-04-07`

不建议直接定义：

- `label = 0` 表示 `final`
- `label = 1` 表示 `added`

原因是：

- 这会把问题收缩成“域来源识别”；
- 但我们真正关心的是“哪条路线的预测更可信”；
- 若后续再出现第三类偏移样本，纯域标签法会迅速失效。

### 4.2 [2026-04-07] 更推荐用“路线收益差”定义 gate 标签

- 状态：`current`
- 首次确认：`2026-04-07`
- 最近复核：`2026-04-07`

更推荐先构造：

- `error_base`
- `error_enhanced`
- `delta = error_base - error_enhanced`

然后定义：

- 若 `delta > 0`
  - 说明增强路线更好
- 若 `delta <= 0`
  - 说明基础路线更好

这时 gate 可以做两种监督：

#### 二分类 gate

- 目标：
  - 预测 `enhanced_better`

#### 连续权重 gate

- 目标：
  - 直接回归一个 `w`
  - 或先回归 `delta`，再映射为权重

这种定义的优点是：

- 不绑定目录名；
- 不绑定某次数据集划分；
- 更贴近真实部署问题。

## 5. gate 应该看什么输入

### 5.1 [2026-04-07] 当前更推荐使用“低成本、可解释、部署时一定能拿到”的特征

- 状态：`current`
- 首次确认：`2026-04-07`
- 最近复核：`2026-04-07`

优先候选特征包括：

- 应变低频能量
- 应变 `3.0-6.0Hz` 能量占比
- 应变高通前后能量差
- 加速度与应变能量比
- FFT `whole` 与 `window` 转速的一致性
- FFT 峰值置信度
- FFT 是否出现固定频带吸附
- 缺失率
- 首尾裁剪比例
- 最长连续缺失长度
- `pred_base` 与 `pred_enhanced` 的差值
- 主干预测的不确定性代理量

当前不建议一开始就把 gate 输入设计成：

- 原始全通道长时序

原因是：

- 当前更需要的是“可解释路由规则”；
- 不是再训练一个更大的黑盒模型。

## 6. 是否必须使用 TCN

### 6.1 [2026-04-07] `midband` 当前语义是“TCN learned 分支的输入前处理”，不是独立解析模块

- 状态：`current`
- 首次确认：`2026-04-07`
- 最近复核：`2026-04-07`

截至 `2026-04-07` 的当前实现里：

- `midband = strain bandpass 3.0-6.0Hz`
- 它是给 `TinyTCN` 这类 learned 分支用的输入前处理；
- 不是替代 FFT 的解析算法。

代码口径上：

- `041` 与 `044` 都是：
  - 先做 `3.0-6.0Hz` 带通；
  - 再把处理后的输入送入 `TinyTCN`；
  - 最后再和解析支线做晚融合。

所以当前“`midband` 在 `final` 上未带来收益”的含义更准确地是：

- `midband + 当前 TinyTCN 分支`
  - 在 `final LOCO` 上尚未显示稳定正增益

而不是：

- `3.0-6.0Hz` 这个频带本身没有信息。

### 6.2 [2026-04-07] 学习 route / gate 时，当前不建议优先继续用 TCN

- 状态：`current`
- 首次确认：`2026-04-07`
- 最近复核：`2026-04-07`

截至 `2026-04-07`，更推荐把 gate 视作：

- 小样本表格判别 / 回归问题

而不是：

- 再训练一个更大的时序深度模型。

原因是：

- 当前最难的不是时序表征能力不足；
- 而是“什么时候该信哪条路线”的条件判断；
- 这种问题在当前样本规模下，更像表格 gate / 路由问题。

## 7. 非 TCN 模型候选优先级

### 7.1 [2026-04-07] 当前 gate 第一优先级应是简单表格模型

- 状态：`current`
- 首次确认：`2026-04-07`
- 最近复核：`2026-04-07`

当前推荐顺序：

1. `LogisticRegression`
2. `LightGBM` / `XGBoost` / `CatBoost`
3. `RandomForest`
4. `MLP`
5. `KNN` / prototype-style route

推荐理由如下。

#### `LogisticRegression`

- 适合先做最小可解释 gate；
- 可以很快看清：
  - 哪些 shift / confidence 特征在驱动路由；
- 适合作为第一条 sanity check 基线。

#### `LightGBM` / `XGBoost` / `CatBoost`

- 更适合学习：
  - 非线性阈值
  - 交互项
  - “某些条件组合下再切路线”的规则；
- 当前很可能是最值得优先尝试的 gate 主候选。

#### `RandomForest`

- 可以作为次级树模型参考；
- 但在这类小样本、规则型问题上，通常不如 boosting 稳。

#### `MLP`

- 可以试；
- 但当前样本规模下，不应作为第一优先级。

#### `KNN / prototype`

- 若后续机制聚类结果稳定，可以把 gate 写成：
  - “更像哪类机制簇”
  - 再决定走哪条路线。

### 7.2 [2026-04-07] `Mixture-of-Experts` 更适合做后续升级，不适合当前第一步

- 状态：`current`
- 首次确认：`2026-04-07`
- 最近复核：`2026-04-07`

后续如果 route / gate 已被证明有效，可以再考虑：

- `Mixture-of-Experts`
- 小型 gating network
- 联合训练结构

但截至 `2026-04-07`，当前不建议一开始就直接做：

- 大一统端到端 gate network

因为：

- 复杂度显著提高；
- 调试难度更大；
- 当前数据量还不足以支撑这类结构成为第一选择。

## 8. 当前推荐的最小执行方案

### 8.1 [2026-04-07] 当前最划算的下一步是 `route/gate quickcheck`

- 状态：`current`
- 首次确认：`2026-04-07`
- 最近复核：`2026-04-07`

建议新增一个轻量 try，目标不是继续扩模型，而是先验证：

- 是否存在一组低成本特征，足以判断：
  - 该走 `base route`
  - 还是该走 `enhanced route`

建议最小版本包括：

#### 路线定义

- `base route`
  - `final` 上更稳的默认路线
- `enhanced route`
  - `added` 上更强的增强路线

#### 特征表

- 只用 case 级可解释特征

#### 模型

- `LogisticRegression`
- `LightGBM` 或 `XGBoost`

#### 比较对象

- 固定走 `base`
- 固定走 `enhanced`
- `gate` 决定 route
- `gate` 输出软权重后的加权融合

#### 验收问题

- gate 是否能把：
  - `final` 上的退化压住
  - 同时保住 `added` 上的增益

## 9. 当前判断

截至 `2026-04-07`，当前更推荐的工程方向是：

- 不直接硬分 `final / added`；
- 而是学习“样本该走哪条路线”的 `route / gate`；
- gate 的第一优先级不应是 `TCN`，而应是：
  - `LogisticRegression`
  - `LightGBM / XGBoost / CatBoost`
- 若 gate 成立，再考虑把它升级为：
  - 软权重融合
  - 或联合结构。

当前最稳妥的表达是：

- `midband` 更像 added 增强模块，而不是 final 默认模块；
- 下一步最值得验证的不是“如何继续放大 midband”，而是“如何自动判断什么时候该用它”。
