# final 训练的 2s+8s 晚融合外推 added2（2026-04-08）

- 状态：`current`
- 首次确认：`2026-04-08`
- 最近复核：`2026-04-08`
- 数据范围：
  - 训练域：`final` 带标签工况 `1,3-20`
  - 外部域：`added` 的 `工况21-24`
  - 外部域：`added2` 的 `工况25-30`
- 代码口径：
  - `src/try/063_final_late_fusion_added2_replay/`
- 证据入口：
  - `outputs/try/063_final_late_fusion_added2_replay/models/checkpoints/`
  - `outputs/try/063_final_late_fusion_added2_replay/case_level_predictions.csv`
  - `outputs/try/063_final_late_fusion_added2_replay/summary_by_domain.csv`
  - `outputs/try/063_final_late_fusion_added2_replay/summary.md`

## 1. 目标

验证一个直接问题：

- 只在 `final` 带标签工况上训练的 `TinyTCN@2s`
- 只在 `final` 带标签工况上训练的 `TinyTCN@8s`
- 以及它们的工况级 `2s+8s` 晚融合

是否可以不做域适配，直接外推到 `added2`。

## 2. 方法口径

- 如果仓库里不存在可复用的 `full final deploy` 权重，则新训练并持久化：
  - `final_deploy_2s.pt`
  - `final_deploy_8s.pt`
- 模型结构与训练口径复用：
  - `src/try/013_phase3_cnn_tcn_smoke/phase3_cnn_tcn_lib.py`
- 窗口口径复用：
  - `2s = 100` 点，步长 `50`
  - `8s = 400` 点，步长 `200`
- 晚融合口径保持与 `026` 一致：
  - `pred_fusion = 0.5 * pred_2s + 0.5 * pred_8s`

## 3. 当前结果

### 3.1 [2026-04-08] `final` 训练的 direct TinyTCN 在 `added2` 上仍然整体高估

`added2` 外部域结果：

- `TinyTCN@2s`
  - `case_mae = 2.1985`
  - `case_rmse = 2.7148`
  - `mean_signed_error = +1.8960`
- `TinyTCN@8s`
  - `case_mae = 2.4157`
  - `case_rmse = 3.0319`
  - `mean_signed_error = +2.3395`
- `TinyTCN 2s+8s late fusion`
  - `case_mae = 2.2897`
  - `case_rmse = 2.8519`
  - `mean_signed_error = +2.1177`

这说明：

- `added2` 虽然比旧 `added` 更接近 `final`，但 direct learned 外推仍然没有变成可部署水平；
- 误差主方向仍然是系统性高估，而不是随机摆动。

### 3.2 [2026-04-08] 在 `added2` 上，`2s+8s` 晚融合没有保住 `final` 域里的优势

`added2` 上三者排序：

- 最优：`2s`
- 次优：`2s+8s fusion`
- 最差：`8s`

这说明：

- `026` 在 `final` 域里成立的“`2s+8s` 明显优于单窗长”的结论，没有自然迁移到 `added2`；
- 对 `added2` 而言，`8s` 长窗反而更容易放大高估；
- `2s+8s` 晚融合在这里更像把 `8s` 的域外偏差带回来了。

### 3.3 [2026-04-08] `added2` 内部同样分成“可外推”和“不可外推”两组

`2s+8s fusion` 的逐工况结果：

- `工况25`
  - `pred = 7.9840`
  - `abs_error = 0.5160`
- `工况26`
  - `pred = 7.9353`
  - `abs_error = 0.2353`
- `工况27`
  - `pred = 8.0363`
  - `abs_error = 1.5363`
- `工况28`
  - `pred = 6.7868`
  - `abs_error = 3.0868`
- `工况29`
  - `pred = 6.8842`
  - `abs_error = 3.2842`
- `工况30`
  - `pred = 8.3797`
  - `abs_error = 5.0797`

这说明：

- `工况25-26` 这类高转速新支路，direct final 模型并不是完全失效；
- `工况28-30` 才是 added2 外推失败的主来源；
- 尤其 `工况30` 继续表现出与旧 added 异常点相似的失败模式。

### 3.4 [2026-04-08] added2 比旧 added 略容易，但仍不足以支撑 direct learned 主线外推

外部域对照：

- `added | 2s`
  - `case_mae = 3.0317`
- `added2 | 2s`
  - `case_mae = 2.1985`
- `added | 2s+8s fusion`
  - `case_mae = 3.5497`
- `added2 | 2s+8s fusion`
  - `case_mae = 2.2897`

这说明：

- `added2` 相比旧 `added` 确实更接近 `final`；
- 但“更接近”还不足以让 `final` 训练的 direct `TinyTCN` 自动变成可靠外推器。

## 4. 当前判断

截至 `2026-04-08`，更合理的判断是：

- `final` 训练的 direct `TinyTCN 2s+8s late fusion` 不能直接升级为 `added2` 默认外推主线；
- 若目标限定为 added2，当前 direct learned 里最多只能把 `2s` 留作参考，不应把 `2s+8s` 当成更优候选；
- `added2` 这批数据更适合：
  - 做 route / gate 校准；
  - 做 reference pool 扩充；
  - 辅助修正 `rpm-first` 主干；
  - 而不是直接验证 “final direct late fusion 已经跨域成立”。

## 5. 一句话版结论

截至 `2026-04-08`，`final` 域里表现最强的 `TinyTCN 2s+8s` 晚融合，并没有自然外推到 `added2`；在 `added2` 上它仍系统性高估，而且还不如单独 `2s`，因此这条线当前不能作为 added2 的默认 direct learned 方案。
