# 032 机制簇内 / 跨簇泛化快速验证

## 目标

- 在不做大规模重实验的前提下，快速验证：
  - 同簇训练是否比跨簇训练更容易泛化；
  - 机制簇对 `TinyTCN@5s` 的泛化是否真的有解释力。

## 设计原则

- 只选一小组代表工况，不跑全量 `19` 折：
  - 难工况：
    - `工况1`
    - `工况3`
    - `工况17`
    - `工况18`
  - 相对容易的代表工况：
    - `工况6`
    - `工况8`
    - `工况15`
    - `工况16`
- 只比较 `TinyTCN@5s`
- 对“同簇 / 跨簇”做 **matched case count**：
  - 每个 eval case 使用相同数量的训练工况；
  - 避免把“簇更像”与“训练样本更多”混在一起。

## 比较对象

- `full`
  - 除 eval case 外的全部带标签工况
- `same_cluster_matched`
  - 只从与 eval case 同簇的训练工况里采样
- `cross_cluster_matched`
  - 只从与 eval case 不同簇的训练工况里采样

## 簇来源

- 使用 `030` 的机制簇结果：
  - `outputs/try/030_case_mechanism_clustering/case_embedding.csv`

## 运行方式

```powershell
.venv\Scripts\python.exe src/try/032_cluster_generalization_quickcheck/run_cluster_generalization_quickcheck.py
```

## 输出

- 输出目录：`outputs/try/032_cluster_generalization_quickcheck/`
- 固定产物：
  - `case_level_results.csv`
  - `summary.csv`
  - `summary.md`
