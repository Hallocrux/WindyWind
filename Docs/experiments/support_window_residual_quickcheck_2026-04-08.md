# support-window residual quickcheck（2026-04-08）

- 状态：`current`
- 首次确认：`2026-04-08`
- 最近复核：`2026-04-08`
- 数据范围：
  - `final` 代表性 holdout：`工况1 / 3 / 17 / 18`
  - `added`：`工况21-24`
- 代码口径：
  - `src/try/053_support_window_residual_quickcheck/`
- 证据入口：
  - `outputs/try/053_support_window_residual_quickcheck/case_level_predictions.csv`
  - `outputs/try/053_support_window_residual_quickcheck/summary_by_domain.csv`
  - `outputs/try/053_support_window_residual_quickcheck/support_window_neighbors.csv`
  - `outputs/try/053_support_window_residual_quickcheck/models/checkpoints/`
  - `outputs/try/053_support_window_residual_quickcheck/summary.md`

## 1. 目标

在 `052` 已经证明 `case-level 2s+8s embedding residual` 有小规模正信号后，继续验证更贴近“参考工况时序支持集”的版本：

- 先用 `rpm_knn4` 找参考工况；
- 不再只做 case-level embedding；
- 而是把参考工况的窗口 embedding 当成 support bank；
- 让目标工况的每个窗口去 support bank 中找近邻窗口，再汇总 residual 修正。

同时控制成本：

- 旧 baseline 全部复用 `052` 已有输出；
- 新训练的 `2s / 8s` 编码器全部持久化 checkpoint，下次复跑直接加载。

## 2. 方法口径

- holdout 工况：
  - `1 / 3 / 17 / 18 / 21 / 22 / 23 / 24`
- 旧对照：
  - 直接读取 `052` 的：
    - `rpm_knn4`
    - `rpm_knn4 + embedding_residual_2s`
    - `rpm_knn4 + embedding_residual_8s`
    - `rpm_knn4 + embedding_residual_concat_2s_8s`
- 新增变体：
  - `rpm_knn4 + support_window_residual_2s`
  - `rpm_knn4 + support_window_residual_8s`
  - `rpm_knn4 + support_window_residual_avg_2s_8s`
  - `rpm_knn4 + support_window_residual_concat_2s_8s`
- `reference cases`：
  - 直接复用 `rpm_knn4` 的 `k=4` 参考工况
- `support-window residual target`：
  - 参考工况对应的 `rpm_residual_oof`
- 权重：
  - 最终修正统一用保守 `w=0.5`

## 3. 当前结果

### 3.1 [2026-04-08] 当前最小版 `support-window residual` 没有超过 `052` 的 case-level concat residual

`focus_all` 对照：

- `052 | rpm_knn4 + embedding_residual_concat_2s_8s @ w=0.5`
  - `case_mae = 0.6006`
- `053 | rpm_knn4 + support_window_residual_2s @ w=0.5`
  - `case_mae = 0.6027`
- `053 | rpm_knn4 + support_window_residual_avg_2s_8s @ w=0.5`
  - `case_mae = 0.6125`
- `053 | rpm_knn4 + support_window_residual_8s @ w=0.5`
  - `case_mae = 0.6268`
- `053 | rpm_knn4 + support_window_residual_concat_2s_8s @ w=0.5`
  - `case_mae = 0.6629`

这说明：

- 把参考工况窗口直接做 support bank，并没有自动优于更简单的 case-level residual；
- 当前最小版 support-window 方向还没有提供额外增益；
- 至少在这组代表性 holdout 上，更复杂的“窗口支持集”并不比 case-level concat 更值得默认保留。

### 3.2 [2026-04-08] `support-window residual` 在 added 上有时更强，但 final 更容易被单个参考工况带偏

`added_focus`：

- `support_window_avg_2s_8s`
  - `case_mae = 0.1343`
- `support_window_2s`
  - `case_mae = 0.1357`
- `052 concat embedding residual`
  - `case_mae = 0.1852`

但 `final_focus`：

- `support_window_concat_2s_8s`
  - `case_mae = 1.0384`
- `support_window_2s`
  - `case_mae = 1.0696`
- `052 concat embedding residual`
  - `case_mae = 1.0159`

这说明：

- support-window 方向在 added 上并不是完全没用，甚至对 `工况21 / 23` 一类样本更强；
- 但它在 final 上更容易被局部 support set 放大偏差；
- 当前更真实的结论是：
  - support-window 增强了“added 修复力”
  - 同时也放大了“final 不稳定性”。

### 3.3 [2026-04-08] `support-window concat` 当前反而是最不稳的 support 版本

结果：

- `support_window_concat_2s_8s | focus_all`
  - `case_mae = 0.6629`
- `support_window_2s | focus_all`
  - `case_mae = 0.6027`
- `support_window_8s | focus_all`
  - `case_mae = 0.6268`

这说明：

- 当前窗口级 concat 并没有复现 `052` 里 case-level concat 的优势；
- 问题更像出在“support set 选择 + 窗口匹配策略”，而不是简单的特征维度不够；
- 当前把窗口级 `2s+8s` 直接拼起来做 support 搜索，还不够稳。

### 3.4 [2026-04-08] `工况21 / 23` 是 support-window 方向最有价值的 added 信号点

代表结果：

- `工况21`
  - `rpm_knn4 = 0.3799`
  - `support_window_avg_2s_8s = 0.0151`
- `工况23`
  - `rpm_knn4 = 0.0420`
  - `support_window_2s = 0.0321`

但同时：

- `工况24`
  - `support_window` 普遍劣于 `052 concat residual`
- `工况22`
  - `support_window` 有改善，但仍弱于 `052 2s residual`

这说明：

- support-window 并不是对所有 added case 都统一更优；
- 它更像对一部分 added-like 样本有强修复力；
- 而对另一部分样本，窗口级 support 反而更容易过拟合参考工况。

## 4. 当前判断

截至 `2026-04-08`，这轮 quickcheck 更支持下面的表达：

- “先用 `rpm_knn4` 找参考工况，再拿参考工况窗口时序做 support” 这个方向不是错的；
- 但当前最小版实现没有超过 `052` 的 case-level concat residual；
- 目前更稳的默认候选仍然是：
  - `rpm_knn4 + case-level embedding residual concat_2s_8s @ w=0.5`

如果还要继续追 support-window，更值得改的不是继续放大训练规模，而是：

- 限制 support set 的 case 组成；
- 加强对 final 的保护；
- 或把 support-window 只开放给少数明显 added-like 的目标样本。

## 5. 一句话版结论

截至 `2026-04-08`，`support-window residual` 已经证明“参考工况时序支持集”这条路对部分 added case 有强信号，但当前最小实现整体上还不如更简单的 case-level `2s+8s` concat residual；它更像一个有潜力的分支，而不是下一步默认升级路线。
