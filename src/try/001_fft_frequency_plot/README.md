# 001 FFT 频谱图探索

## 目标

- 先把“时域信号做 FFT 后的频域图像”跑通并固定为一个可复现入口。
- 当前默认展示一个传感器通道的单边幅值谱，方便后续继续做主频识别、PSD 对比和多工况比较。

## 默认口径

- 数据源：`data/final/datasets/`
- 默认工况：`工况1`
- 默认通道：`WSMS00001.AccX`
- 清洗逻辑：直接复用 `src/current/data_loading.py`
- 频谱算法：
  - 从清洗后的整段时域信号中取该通道；
  - 减去均值，避免直流分量主导；
  - 使用 `numpy.fft.rfft` 计算单边 FFT；
  - 输出单边幅值谱，并限制显示到 Nyquist 频率。

## 运行方式

```powershell
uv run python src/try/001_fft_frequency_plot/plot_fft_spectrum.py
```

也可以显式指定工况编号、通道和输出文件：

```powershell
uv run python src/try/001_fft_frequency_plot/plot_fft_spectrum.py --case-id 4 --column WSMS00003.AccZ --output outputs/try/001_fft_frequency_plot/case4_accz_fft.png
```

## 输出

- 默认图片输出到：`outputs/try/001_fft_frequency_plot/case1_WSMS00001_AccX_fft.png`
- 图片内容包含：
  - 上半部分：原始时域波形；
  - 下半部分：FFT 后的频域单边幅值谱。
