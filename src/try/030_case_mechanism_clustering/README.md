# 030 工况机制聚类探索

## 目标

- 验证“工况并不是同分布一团，而可能存在几类不同机制”这一直觉；
- 先只做第一层：
  - 构建 `per-case mechanism table`
  - 做低维投影与聚类
  - 把误差、窗长偏好等信息叠到可视化上
- 不引入新模型训练。

## 输入与口径

- 数据入口：
  - `data/final/dataset_manifest.csv`
  - `data/final/datasets/工况*.csv`
- 清洗逻辑：复用 `src/current.data_loading`
- 窗口特征：复用 `src/current.features`
- 默认窗口参数：
  - `50Hz`
  - `5s`
  - `2.5s`

## 机制特征来源

- 数据质量：
  - 缺失率
  - 边界裁剪比例
  - 窗口数
  - 时长
- 应变侧聚合统计
- 加速度侧聚合统计
- 应变/加速度相对比值

## 注记但不进入聚类的字段

- 风速标签
- RPM 标签
- `TinyTCN@5s` 的 `LOCO` 工况误差
- 风速最优窗长
- RPM 最优窗长

## 运行方式

```powershell
.venv\Scripts\python.exe src/try/030_case_mechanism_clustering/run_case_mechanism_clustering.py
```

## 输出

- 输出目录：`outputs/try/030_case_mechanism_clustering/`
- 固定产物：
  - `case_mechanism_table.csv`
  - `case_embedding.csv`
  - `cluster_summary.csv`
  - `pca_cluster_scatter.png`
  - `mechanism_heatmap.png`
  - `summary.md`
