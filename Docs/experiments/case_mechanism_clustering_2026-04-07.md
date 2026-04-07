# 工况机制聚类探索（2026-04-07）

- 状态：`current`
- 首次确认：`2026-04-07`
- 最近复核：`2026-04-07`
- 数据范围：`工况1` 到 `工况20`
- 代码口径：
  - `src/try/030_case_mechanism_clustering/`
- 证据入口：
  - `outputs/try/030_case_mechanism_clustering/case_mechanism_table.csv`
  - `outputs/try/030_case_mechanism_clustering/case_embedding.csv`
  - `outputs/try/030_case_mechanism_clustering/cluster_summary.csv`
  - `outputs/try/030_case_mechanism_clustering/pca_cluster_scatter.png`
  - `outputs/try/030_case_mechanism_clustering/mechanism_heatmap.png`
  - `outputs/try/030_case_mechanism_clustering/summary.md`

## 1. 目标

验证“工况并不是同分布一团，而可能存在几类不同机制”这一工作假设的第一层证据：

- 构建 `per-case mechanism table`
- 做低维投影与聚类
- 把风速误差和窗长偏好叠到可视化上

## 2. 方法口径

- 聚类特征只使用机制特征，不直接使用：
  - 风速标签
  - RPM 标签
  - 模型误差
  - 最优窗长
- 机制特征包括：
  - 数据质量
  - 时长 / 窗口数
  - 应变侧聚合统计
  - 加速度侧聚合统计
  - 应变 / 加速度相对比值
- 聚类方法：
  - 标准化后做 `AgglomerativeClustering`
  - 在 `k=2/3/4` 中按 silhouette 选最优
- 可视化：
  - PCA 散点图
  - 机制特征热力图

## 3. 当前结果

### 3.1 [2026-04-07] 当前 `20` 工况的第一层机制聚类更支持 `2` 类结构

- 状态：`current`
- 首次确认：`2026-04-07`
- 最近复核：`2026-04-07`

当前结果：

- 选定簇数：`2`
- silhouette score：`0.3276`
- PCA explained variance：
  - `PC1 = 37.09%`
  - `PC2 = 24.66%`

两类工况分配为：

- `cluster 0`
  - `1,2,3,4,5,6,7,8,9,10,11,12,13,14,19,20`
- `cluster 1`
  - `15,16,17,18`

### 3.2 [2026-04-07] 当前最主要的机制分裂由 `15-18` 这一簇驱动

- 状态：`current`
- 首次确认：`2026-04-07`
- 最近复核：`2026-04-07`

`cluster 1` 相比 `cluster 0` 的主要特征是：

- `acc_energy_median` 更高
- `acc_std_median` 更高
- `strain_rms_median` 更低
- `strain_acc_rms_ratio` 更低
- 缺失率与边界裁剪比例也更高

这说明：

- `15 / 16 / 17 / 18` 不只是标签更高或编号更新；
- 它们在“加速度侧更强、应变相对更弱、质量口径也更不同”这一组机制特征上，确实形成了一个单独簇。

### 3.3 [2026-04-07] 高误差工况并没有全部集中到同一个簇

- 状态：`current`
- 首次确认：`2026-04-07`
- 最近复核：`2026-04-07`

风速 `TinyTCN@5s` 的高误差工况包括：

- `工况1`
  - `cluster 0`
  - `wind_loco_error_5s = 1.1989`
- `工况3`
  - `cluster 0`
  - `wind_loco_error_5s = 1.0515`
- `工况17`
  - `cluster 1`
  - `wind_loco_error_5s = 1.0883`
- `工况18`
  - `cluster 1`
  - `wind_loco_error_5s = 0.6433`

这说明：

- “难工况”不是全部来自同一个机制簇；
- 至少存在两类不同的困难来源：
  - `cluster 0` 内的低风速 / 早期工况型困难
  - `cluster 1` 内的高能量 / 新增工况型困难

## 4. 当前判断

`2026-04-07` 的第一层探索支持以下判断：

- 工况确实不是完全同分布的一团；
- 当前最明显的第一层机制分裂，是 `15-18` 相对其余工况形成单独簇；
- 但“高误差工况”并没有集中在一个簇里，因此下一步不能只做“把异常簇单独剥离”；
- 更合理的后续方向是：
  - 继续做第二层“误差模式聚类”
  - 或在当前簇划分下做“簇内训练 / 跨簇验证”。
