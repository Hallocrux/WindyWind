# case evidence aggregator quickcheck（2026-04-08）

- 状态：`current`
- 首次确认：`2026-04-08`
- 最近复核：`2026-04-08`
- 数据范围：
  - `final` 代表性 holdout：`工况1 / 3 / 17 / 18`
  - `added`：`工况21-24`
- 代码口径：
  - `src/try/054_case_evidence_aggregator_quickcheck/`
- 证据入口：
  - `outputs/try/054_case_evidence_aggregator_quickcheck/evidence_table.csv`
  - `outputs/try/054_case_evidence_aggregator_quickcheck/case_level_predictions.csv`
  - `outputs/try/054_case_evidence_aggregator_quickcheck/summary_by_domain.csv`
  - `outputs/try/054_case_evidence_aggregator_quickcheck/models/`
  - `outputs/try/054_case_evidence_aggregator_quickcheck/summary.md`

## 1. 目标

验证一种更符合“窗口给证据，case 做决策”的表达：

- 不再给窗口直接分配 case-level 常数 residual；
- 直接复用 `052 / 053` 已经落盘的 case-level correction 候选；
- 再结合 `047` 已有的 case-level 机制特征；
- 在 case 层预测：
  - `residual = true_wind_speed - rpm_knn4_base`

## 2. 方法口径

- 复用输入：
  - `052` 的 case-level embedding residual 候选
  - `053` 的 support-window residual 候选
  - `047` 的 gate feature table
- 目标：
  - `residual_target = true_wind_speed - base_pred`
- 小模型：
  - `ridge_candidate_only`
  - `ridge_candidate_plus_mechanism`
  - `ridge_consensus_plus_mechanism`
- 评估：
  - 代表性 holdout 上的 case-level `LOOCV`
- 持久化：
  - 每个 holdout fold 的 ridge pipeline 全部落盘到 `models/`

## 3. 当前结果

### 3.1 [2026-04-08] 当前 case-level evidence aggregator 没有超过 `052` 的固定最佳 residual 候选

`focus_all` 对照：

- `052 | rpm_knn4 + embedding_residual_concat_2s_8s @ w=0.5`
  - `case_mae = 0.6006`
- `054 | ridge_candidate_only @ w=0.5`
  - `case_mae = 0.7477`
- `054 | ridge_candidate_plus_mechanism @ w=0.5`
  - `case_mae = 0.9150`
- `054 | ridge_consensus_plus_mechanism @ w=0.5`
  - `case_mae = 0.9032`

这说明：

- 把已有 correction 候选再交给一个 case-level ridge aggregator，并没有自动把信号“洗干净”；
- 当前更稳的还是 `052` 里已经筛出来的固定候选；
- 至少在这组代表性 holdout 上，evidence aggregation 这一步还没有形成额外收益。

### 3.2 [2026-04-08] 只用 correction 候选做小幅 shrink 的 `ridge_candidate_only__w0.5` 是当前最接近可保留的 aggregator 版本

`focus_all`：

- `rpm_knn4 = 0.6444`
- `ridge_candidate_only__w0.5 = 0.7477`

虽然没超过 `rpm_knn4`，但相对其他 aggregator 版本更稳。

这说明：

- 当前如果还要继续追 case-level aggregator，最值得保留的是：
  - 少特征
  - 强收缩
  - 不要一上来混太多机制特征

### 3.3 [2026-04-08] 机制特征在这组极小样本上没有带来稳定增益，反而更容易把 hard case 拉偏

代表结果：

- `工况17`
  - `ridge_candidate_plus_mechanism__w0.5`
    - `pred = 5.8611`
    - `abs_error = 2.2389`
- `工况22`
  - `ridge_candidate_plus_mechanism__w0.5`
    - `pred = 2.3637`
    - `abs_error = 1.0363`

这说明：

- 机制特征本身不是没信息；
- 但在当前 `8` 个 holdout 的极小样本 quickcheck 下，它们更容易被小模型过度解释；
- 当前主要问题不是“机制特征完全不对”，而是“样本太小，加入机制特征后自由度太高”。

## 4. 当前判断

截至 `2026-04-08`，这轮 quickcheck 更支持下面的表达：

- “窗口给证据，case 做修正” 这个方向在问题定义上是对的；
- 但当前这版 case-level ridge aggregator 还没有把信号提纯到优于 `052` 最佳固定候选；
- 所以它暂时更像一个分析工具，而不是下一步默认升级路线。

如果还要继续追 case-level aggregation，更值得改的方向不是继续加特征，而是：

- 降低自由度；
- 改成更保守的 gate / bucket；
- 或只预测“是否允许启用某个已知 correction”，而不是直接回归连续 residual。

## 5. 一句话版结论

截至 `2026-04-08`，case-level evidence aggregator 已经证明“问题应该在 case 层解决”这一点没有错，但当前最小版 ridge 聚合还没超过 `052` 的固定 `2s+8s` concat residual；下一步更值得做的是保守 gate，而不是继续扩大连续 residual 回归。
