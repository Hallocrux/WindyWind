# `pred_rpm` 可部署性验证（2026-04-07）

- 状态：`current`
- 首次确认：`2026-04-07`
- 最近复核：`2026-04-07`
- 数据范围：
  - `final` 旧域：`data/final/` 的带标签工况
  - `added` 外部域：`data/added/` 的 `工况21-24`
- 代码口径：
  - `src/try/043_pred_rpm_deployability_check/`
- 证据入口：
  - `outputs/try/043_pred_rpm_deployability_check/rpm_case_level_predictions.csv`
  - `outputs/try/043_pred_rpm_deployability_check/rpm_to_wind_case_level_predictions.csv`
  - `outputs/try/043_pred_rpm_deployability_check/rpm_to_wind_summary.csv`
  - `outputs/try/043_pred_rpm_deployability_check/deployable_vs_true_rpm_gap.csv`
  - `outputs/try/043_pred_rpm_deployability_check/summary.md`

## 1. 目标

把 `2026-04-07` 计划书里的第一个 try 正式落地，回答：

- 真实 `rpm` 不可用时，`pred_rpm -> rpm_to_wind` 是否还能保留 added 增益；
- `rpm_knn4` 是否过度依赖精确 `rpm`；
- 是否存在更平滑的 `rpm -> wind` 映射，可以容忍 `pred_rpm` 误差。

## 2. 方法口径

- rpm 预测主干：
  - 复用 `019/024` 的 TinyTCN rpm 回归口径
- 清洗与切窗：
  - 复用 `src/current.data_loading`
  - 复用 `src/try/012_phase3_end_to_end_shortlist/phase3_end_to_end_lib.py`
- `final` 旧域评估：
  - `LOCO`
  - `2.0s / 3.0s / 5.0s` 的 case 级 rpm 预测优先复用 `024` 已落盘结果
- `added` 外部评估：
  - 用全部 `final` 带标签工况训练 rpm 模型
  - 对 `工况21-24` 做外部推理
- `rpm -> wind` 映射：
  - `rpm_knn4`
  - `rpm_linear`
  - `ridge_rpm_to_wind`
    - 当前实现为 `rpm + rpm^2` 的二次 `Ridge` 平滑映射

## 3. 变体矩阵

- `true_rpm -> rpm_knn4`
- `true_rpm -> rpm_linear`
- `true_rpm -> ridge_rpm_to_wind`
- `pred_rpm_3.0s -> rpm_knn4`
- `pred_rpm_3.0s -> rpm_linear`
- `pred_rpm_3.0s -> ridge_rpm_to_wind`
- `pred_rpm_2.0s -> rpm_linear`
- `pred_rpm_5.0s -> rpm_linear`

## 4. 当前结果

### 4.1 [2026-04-07] `pred_rpm` 支线在 `final LOCO` 上仍可工作

- 状态：`current`
- 首次确认：`2026-04-07`
- 最近复核：`2026-04-07`

`final LOCO` 最优可部署链为：

- `pred_rpm_3.0s -> ridge_rpm_to_wind`
  - `case_mae = 0.3262`
  - `case_rmse = 0.4487`

对照上界：

- `true_rpm -> ridge_rpm_to_wind`
  - `case_mae = 0.3430`
- `true_rpm -> rpm_linear`
  - `case_mae = 0.4063`
- `true_rpm -> rpm_knn4`
  - `case_mae = 0.4256`

这说明：

- 在 `final` 旧域里，`pred_rpm` 本身虽然仍有 case 级误差，但通过平滑映射后还能落回可接受范围；
- 当前 `pred_rpm` 支线的问题，不是“在所有域都不能用”，而是“域外稳定性不足”。

### 4.2 [2026-04-07] `pred_rpm` 支线在 `added` 外部域上发生整体高估崩坏

- 状态：`current`
- 首次确认：`2026-04-07`
- 最近复核：`2026-04-07`

`added` 外部域的 case 级 rpm 误差为：

- `TinyTCN@2.0s`
  - `rpm_mae = 74.6552`
  - `rpm_max_abs = 146.3480`
- `TinyTCN@3.0s`
  - `rpm_mae = 77.5826`
  - `rpm_max_abs = 152.2184`
- `TinyTCN@5.0s`
  - `rpm_mae = 93.2017`
  - `rpm_max_abs = 171.2661`

典型外推结果显示：

- `工况21`
  - 真值 `168 rpm`
  - 预测约 `255-274 rpm`
- `工况22`
  - 真值 `106 rpm`
  - 预测约 `252-277 rpm`
- `工况24`
  - 真值 `204 rpm`
  - 预测约 `253-275 rpm`

这说明：

- 当前 `pred_rpm` 模型在 added 上不是“略有偏差”，而是出现了明显向高转速区间塌缩的域外失配；
- 因此后续若继续保留解析支线，首要问题不再是 `rpm -> wind` 映射细节，而是 `pred_rpm` 本身的外部泛化。

### 4.3 [2026-04-07] added 上不存在满足通过条件的 `pred_rpm -> wind` 方案

- 状态：`current`
- 首次确认：`2026-04-07`
- 最近复核：`2026-04-07`

`added` 外部域结果：

- 上界参考
  - `true_rpm -> rpm_knn4`
    - `case_mae = 0.2293`
  - `true_rpm -> ridge_rpm_to_wind`
    - `case_mae = 0.3614`
- 最优可部署链
  - `pred_rpm_2.0s -> rpm_linear`
    - `case_mae = 1.8886`
- `3.0s` 主候选
  - `pred_rpm_3.0s -> rpm_linear`
    - `case_mae = 1.9719`
  - `pred_rpm_3.0s -> ridge_rpm_to_wind`
    - `case_mae = 2.1123`
  - `pred_rpm_3.0s -> rpm_knn4`
    - `case_mae = 2.4701`

相对当前纯 learned 中频参考：

- `TinyTCN all_channels midband`
  - `case_mae = 0.2926`

这说明：

- 当前没有任何 `pred_rpm -> wind` 方案在 added 上接近 `true_rpm` 上界；
- 当前没有任何 `pred_rpm -> wind` 方案优于纯 learned 中频支线；
- `rpm_knn4` 对 noisy `pred_rpm` 最敏感；
- 更平滑的 `rpm_linear / ridge` 虽然比 `knn4` 稍稳，但仍远未达到可保留水平。

## 5. 当前判断

`2026-04-07` 的 `043` 已经触发计划书里的失败回退条件：

- `C = pred_rpm` 当前只能保留为研究支线；
- `044` 的默认统一融合矩阵不应继续把 `pred_rpm` 当作主候选辅助支线；
- 下一步若继续做 `044`，默认应优先先做：
  - `A`
  - `A + B`
  - 仅把 `A + C` 作为次级补充参考

更具体地说：

- `final` 域内的解析支线并没有完全失效；
- 但当前 added 的主矛盾已经明确落在 `pred_rpm` 外部域泛化，而不是 `rpm_to_wind` 映射形式；
- 因此若要让 `C` 重新回到候选矩阵，后续需要优先补的是：
  - `pred_rpm` 的 added 域适配 / 校准
  - 或显式的解析支线外部域校正
