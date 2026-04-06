# 006 工况5 中间非缺失部分逐列图像

## 目标

- 把 `工况5` 删除首尾连续缺失段后的“中间非缺失部分”按列画出来。
- 方便直接观察每个有效通道在稳定保留段中的波形形态、幅值量级和大致规律。

## 输入与口径

- 数据文件：`data/final/datasets/工况5.csv`
- 清洗逻辑：复用 `src/current/data_loading.py`
- 当前展示口径：
  - 先删除首尾连续缺失段；
  - 中间部分保留，沿用当前主线的缺失填充结果；
  - 展示全部 `20` 个共有有效信号列。

## 输出

- 输出目录：`outputs/try/006_case5_middle_segment_plots/`
- 主要产物：
  - `case5_trimmed_all_channels_small_multiples.png`
    - 20 个有效信号列的逐列小图
  - `case5_trimmed_acceleration_overlay.png`
    - 全部加速度通道的归一化总览图
  - `case5_trimmed_strain_overlay.png`
    - 全部应变通道的归一化总览图
  - `channel_summary.csv`
    - 每列的均值、标准差、最值等摘要
  - `summary.md`

## 运行方式

```powershell
uv run python src/try/006_case5_middle_segment_plots/plot_case5_middle_segment.py
```
