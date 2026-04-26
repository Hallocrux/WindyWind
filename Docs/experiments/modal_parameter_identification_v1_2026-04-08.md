# 模态参数识别探索线 v1 实现与 smoke 记录（2026-04-08）

- 状态：`historical`
- 首次确认：`2026-04-08`
- 最近复核：`2026-04-09`
- 替代关系：
  - `2026-04-09` 起，默认主入口已迁移到 `src/modal_parameter_identification/`
  - `2026-04-09` 起，默认输出目录已迁移到 `outputs/modal_parameter_identification/`
- 数据范围：
  - smoke 工况：`工况1`、`工况10`、`工况17`
- 代码口径：
  - `src/current.data_loading`
  - `src/try/065_modal_parameter_identification/`
- 证据入口：
  - `outputs/try/065_modal_parameter_identification/case_modal_summary.csv`
  - `outputs/try/065_modal_parameter_identification/stability_statistics.csv`
  - `outputs/try/065_modal_parameter_identification/case_01_modal_overview.png`
  - `outputs/try/065_modal_parameter_identification/case_10_modal_overview.png`
  - `outputs/try/065_modal_parameter_identification/case_17_modal_overview.png`

## 1. 本次实现内容

`2026-04-08` 已新增一条独立探索线，用于把模态参数识别工作流落成可复现脚本，当前已实现：

- `PSD / CSD / coherence / FDD` 频域预分析
- `1x-4x` 谐波标记与屏蔽表输出
- `SSI-COV` 低频主模态识别
- `strain` 与 `acc_y` 两套 `5` 测点离散振型输出
- 多窗口稳定性统计
- 可选 FE 参考文件读取与对齐接口

## 2. 当前 smoke 结果

`2026-04-08` 在 `工况1`、`工况10`、`工况17` 上的 smoke 输出如下：

| case_id | strain_first_frequency_hz | strain_damping_ratio | accy_first_frequency_hz | accy_damping_ratio |
| --- | ---: | ---: | ---: | ---: |
| 1 | 2.3623 | 0.0030 | 2.3583 | 0.0043 |
| 10 | 2.1903 | 0.0117 | 2.4250 | 0.0222 |
| 17 | 2.1787 | 0.0083 | 2.3500 | 0.0228 |

对应多窗口稳定性摘要：

- `工况1`
  - `strain valid_window_count = 5`
  - `acc_y valid_window_count = 5`
  - `strain / acc_y` 频率差约 `0.0041 Hz`
- `工况10`
  - `strain valid_window_count = 12`
  - `acc_y valid_window_count = 12`
  - `strain / acc_y` 频率差约 `0.2347 Hz`
- `工况17`
  - `strain valid_window_count = 11`
  - `acc_y valid_window_count = 11`
  - `strain / acc_y` 频率差约 `0.1713 Hz`

## 3. 当前判断

### 3.1 [2026-04-08] 第一版实现已经能稳定输出完整工件链

- 对 smoke 工况，当前脚本已能稳定生成：
  - 工况级摘要
  - 窗口级模态估计
  - 谐波标记表
  - 稳定极点表
  - `strain / acc_y` 振型
  - 工况图像概览

这说明第一版工程落点已经形成，不再只是计划。

### 3.2 [2026-04-08] `strain` 侧在 smoke 工况上比 `acc_y` 更稳

- `工况1` 上两者高度一致；
- `工况10` 与 `工况17` 上，`acc_y` 的窗口频率离散度高于 `strain`；
- `strain` 的 `frequency_std_hz` 在当前 smoke 中整体更小。

这说明当前默认把 `strain` 作为主参考、`acc_y` 作为辅助对照是合理的。

## 4. 当前限制

- `2026-04-08` 当前阻尼实现仍是 `EFDD-like` 工程近似，不应直接当成最终高精度阻尼结论。
- `2026-04-08` 当前 `SSI-COV` 还没有扩展到多模态自动筛选与跨工况统一门限整定。
- `2026-04-08` 当前 smoke 只覆盖 `3` 个代表工况，尚未形成 `工况1-20` 的全量稳定结论。
- `2026-04-08` 当前 FE 对比只完成了标准化接口，仓库内尚未附带 FE 参考文件实例。

## 5. 下一步建议

- 把 `工况1-20` 全量跑通，复核 `2.2-2.4 Hz` 候选带在 `strain / acc_y` 双口径下的稳定性。
- 增加更保守的 `SSI` 稳定极点筛选与 cluster 规则，降低 `acc_y` 在部分工况上的频率离散度。
- 若后续提供同步 rpm 时程，再复核“工况级常值 rpm”与“窗口级同步 rpm”两种谐波屏蔽口径的差异。
