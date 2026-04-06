# TinyTCN 边界窗口误差检查（2026-04-06）

- 状态：`current`
- 首次确认：`2026-04-06`
- 最近复核：`2026-04-06`
- 数据范围：
  - 风速任务带标签工况中的 `工况1 / 3 / 17 / 18`
- 代码口径：
  - `src/try/025_tinytcn_boundary_error_check/`
- 证据入口：
  - `outputs/try/025_tinytcn_boundary_error_check/target_case_segment_error_summary.csv`
  - `outputs/try/025_tinytcn_boundary_error_check/target_case_segment_tests.csv`
  - `outputs/try/025_tinytcn_boundary_error_check/summary.md`

## 1. 目标

在 `2026-04-06` 当前 `TinyTCN + 50Hz / 5s / 2.5s + LOCO` 口径下，专门检查：

- `工况1 / 3 / 17 / 18`
- 不同窗口位置的预测误差是否显著不同；
- “边界段有害”是否能被更直接的证据支持。

## 2. 方法口径

- 模型：`TinyTCN`
- 评估方式：
  - `Leave-One-Condition-Out`
  - 只重跑目标 `4` 个 eval 工况折
- 位置分段：
  - `start`：开始 `15s`
  - `middle`：中间 `15s`
  - `end`：结束 `15s`
- 统计比较：
  - `start vs middle`
  - `end vs middle`
  - `boundary(start+end) vs middle`
- 显著性检验：
  - 比较平均绝对误差差值
  - 单侧置换检验

## 3. 当前结果

### 3.1 [2026-04-06] `工况1` 与 `工况18` 支持“边界段误差更大”

- 状态：`current`
- 首次确认：`2026-04-06`
- 最近复核：`2026-04-06`

`boundary vs middle` 结果：

- `工况1`
  - `boundary_mean_abs_error = 1.7805`
  - `middle_mean_abs_error = 0.7158`
  - `p = 0.0056`
- `工况18`
  - `boundary_mean_abs_error = 0.6683`
  - `middle_mean_abs_error = 0.2211`
  - `p = 0.0140`

补充细节：

- `工况1` 的结束段最差：
  - `end_mean_abs_error = 2.9281`
  - `end vs middle p = 0.0143`
- `工况18` 的开始段最差：
  - `start_mean_abs_error = 0.8925`
  - `start vs middle p = 0.0079`

### 3.2 [2026-04-06] `工况3` 与 `工况17` 不支持“边界段有害”这一解释

- 状态：`current`
- 首次确认：`2026-04-06`
- 最近复核：`2026-04-06`

`boundary vs middle` 结果：

- `工况3`
  - `boundary_mean_abs_error = 0.8589`
  - `middle_mean_abs_error = 1.3150`
  - `p = 0.9916`
- `工况17`
  - `boundary_mean_abs_error = 1.1002`
  - `middle_mean_abs_error = 1.1429`
  - `p = 0.6853`

这说明：

- `工况3` 的误差更像是中段整体偏高，而不是边界段单独拖后腿；
- `工况17` 的三段误差接近，边界段不是主要解释。

## 4. 当前判断

`2026-04-06` 的这轮检查支持以下判断：

- “边界段系统性有害”不能直接外推到所有高误差工况；
- 但在 `工况1` 与 `工况18` 上，边界段更差已经有比较直接的统计证据；
- 如果继续沿用 `TinyTCN` 主线，值得继续验证“稳态窗口优先 / 边界降权”。
