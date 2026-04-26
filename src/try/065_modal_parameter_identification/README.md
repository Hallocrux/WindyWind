# 065 模态参数识别探索线

- 状态：`historical`
- 首次确认：`2026-04-08`
- 最近复核：`2026-04-09`
- 替代关系：
  - `2026-04-09` 起默认主入口迁移到 `src/modal_parameter_identification/`
  - `2026-04-09` 起默认输出目录迁移到 `outputs/modal_parameter_identification/`

`2026-04-09` 起，这个目录只保留历史实现与兼容脚本；后续默认使用正式模块：

```powershell
uv run python -m src.modal_parameter_identification
```

## 目标

- 在 `2026-04-08` 当前仓库口径下，建立一条可复现的模态参数识别探索线。
- 覆盖以下工作流：
  - 多通道时程读取
  - `PSD / CSD / coherence / FDD` 预分析
  - `1P-4P` 谐波标记与屏蔽
  - 一阶 `tower bending` 候选提取
  - `SSI-COV` 主识别
  - `strain` 与 `acc_y` 两套 `5` 测点离散振型
  - 多窗口稳定性统计
  - 可选 `FE / 梁模型` 对比

## 输入数据

- `data/final/dataset_manifest.csv`
- `data/final/datasets/工况{ID}.csv`
- 可选同步 rpm 时程 CSV：
  - 必要列：`time`、`rpm`
  - 可选列：`case_id`
- 可选 FE 参考 CSV：
  - `basis`
  - `mode_label`
  - `frequency_hz`
  - `damping_ratio`
  - `point_1` 到 `point_5`

## 方法口径

- 代码口径：
  - `src/current/data_loading.py`
  - `src/try/065_modal_parameter_identification/`
- 默认采样率：`50 Hz`
- 默认预分析频带：`0.5-6.0 Hz`
- 默认关注频带：`2.0-3.0 Hz`
- 默认 `SSI` 窗口：`20 s`
- 默认步长：`10 s`
- 默认谐波屏蔽：`1x-4x`
- 默认谐波半宽：`±0.2 Hz`
- 默认加速度振型口径：
  - 只输出 `AccY` 的 `5` 测点离散振型
  - `AccX / AccZ` 只保留在诊断层，不进入默认振型输出

## 运行方式

默认同时输出 `strain + acc_y`：

```powershell
uv run python src/try/065_modal_parameter_identification/run_modal_parameter_identification.py
```

只跑部分工况：

```powershell
uv run python src/try/065_modal_parameter_identification/run_modal_parameter_identification.py `
  --case-ids 1 10 17
```

指定同步 rpm 时程：

```powershell
uv run python src/try/065_modal_parameter_identification/run_modal_parameter_identification.py `
  --rpm-source sync_csv `
  --rpm-series-path path/to/rpm_series.csv
```

指定 FE 参考：

```powershell
uv run python src/try/065_modal_parameter_identification/run_modal_parameter_identification.py `
  --fe-reference-path path/to/fe_reference.csv
```

## 输出位置

- `outputs/try/065_modal_parameter_identification/case_modal_summary.csv`
- `outputs/try/065_modal_parameter_identification/window_modal_estimates.csv`
- `outputs/try/065_modal_parameter_identification/harmonic_mask_table.csv`
- `outputs/try/065_modal_parameter_identification/stabilization_poles.csv`
- `outputs/try/065_modal_parameter_identification/stability_statistics.csv`
- `outputs/try/065_modal_parameter_identification/strain_mode_shapes.csv`
- `outputs/try/065_modal_parameter_identification/accy_mode_shapes.csv`
- `outputs/try/065_modal_parameter_identification/case_*_modal_overview.png`
- `outputs/try/065_modal_parameter_identification/case_*_mode_shape_comparison.png`
- `outputs/try/065_modal_parameter_identification/fe_comparison.csv`
  - 仅在提供 FE 参考时生成

## 当前限制

- `2026-04-08` 当前正式数据默认只有工况级 `rpm`，没有仓库内置的逐时刻同步 rpm 时程示例文件。
- `2026-04-08` 当前阻尼估计采用基于奇异值峰宽的 `EFDD-like` 简化实现，更适合作为第一版工程近似，而不是最终精细阻尼口径。
- `2026-04-08` 当前 `SSI-COV` 以低频一阶候选为主，不面向多模态全自动稳定图整定。
- `2026-04-08` 当前 `acc` 振型默认固定为 `AccY`，尚未引入基于坐标系或安装方向的物理映射文件。
