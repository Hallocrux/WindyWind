# 005 工况5 RPM 频域诊断

## 目标

- 针对 `工况5.csv`，检查频域图中是否能看到与 `166 rpm` 对应的频率特征。
- 当前重点不是重新训练模型，而是做可视化诊断：
  - 看 `1x rpm = 166 / 60 = 2.7667 Hz` 附近是否存在明显峰值；
  - 看 `2x / 3x / 4x` 倍频是否更清晰；
  - 给出各加速度通道在这些目标频率处的响应强度。

## 输入与口径

- 数据文件：`data/final/datasets/工况5.csv`
- 清洗逻辑：复用 `src/current/data_loading.py`
- 本次使用“稳定保留段”口径：
  - 删除首尾连续缺失段；
  - 中间部分保留并使用当前主线的填充结果；
  - 对裁剪后的整段信号直接做频域分析。
- 重点分析对象：
  - 所有加速度通道 `WSMS*`
  - 重点标注目标频率：
    - `1x rpm = 2.7667 Hz`
    - `2x rpm = 5.5333 Hz`
    - `3x rpm = 8.3000 Hz`
    - `4x rpm = 11.0667 Hz`

## 输出

- 输出目录：`outputs/try/005_case5_rpm_frequency_scan/`
- 主要产物：
  - `case5_trimmed_acc_overview.png`
    - 所有加速度通道的频域总览图，并标出 rpm 及倍频位置
  - `case5_trimmed_selected_channels.png`
    - 若干重点通道的频谱细图
  - `case5_frequency_targets.csv`
    - 各通道在目标频率处的幅值
  - `summary.md`
    - 当前人工结论摘要

## 运行方式

```powershell
uv run python src/try/005_case5_rpm_frequency_scan/analyze_case5_rpm_frequency.py
```
