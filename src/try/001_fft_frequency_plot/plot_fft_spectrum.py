from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.current.data_loading import (
    get_common_signal_columns,
    load_clean_signal_frame,
    scan_dataset_records,
)


DEFAULT_CASE_ID = 1
DEFAULT_COLUMN = "WSMS00001.AccX"
DEFAULT_OUTPUT = Path("outputs/try/001_fft_frequency_plot/case1_WSMS00001_AccX_fft.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="绘制时域信号 FFT 后的频谱图。")
    parser.add_argument("--case-id", type=int, default=DEFAULT_CASE_ID, help="工况编号。")
    parser.add_argument(
        "--column",
        default=DEFAULT_COLUMN,
        help="要分析的传感器通道名，例如 WSMS00001.AccX。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="输出图片路径。",
    )
    return parser.parse_args()


def select_record(case_id: int):
    records = scan_dataset_records()
    for record in records:
        if record.case_id == case_id:
            return record, records
    raise ValueError(f"未找到工况 {case_id} 对应的数据文件。")


def estimate_sampling_rate(time_seconds: np.ndarray) -> float:
    deltas = np.diff(time_seconds)
    if len(deltas) == 0:
        raise ValueError("采样点数量不足，无法估计采样频率。")

    median_dt = float(np.median(deltas))
    if median_dt <= 0:
        raise ValueError(f"采样时间间隔异常: {median_dt}")
    return 1.0 / median_dt


def compute_single_sided_spectrum(signal: np.ndarray, sampling_rate: float) -> tuple[np.ndarray, np.ndarray]:
    centered = signal - np.mean(signal)
    fft_values = np.fft.rfft(centered)
    frequencies = np.fft.rfftfreq(len(centered), d=1.0 / sampling_rate)
    amplitudes = (2.0 / len(centered)) * np.abs(fft_values)
    if len(amplitudes) > 0:
        amplitudes[0] *= 0.5
    return frequencies, amplitudes


def main() -> None:
    args = parse_args()
    record, records = select_record(args.case_id)
    common_signal_columns = get_common_signal_columns(records)

    if args.column not in common_signal_columns:
        available = ", ".join(common_signal_columns)
        raise ValueError(f"通道 {args.column} 不在共有有效通道中。可选通道: {available}")

    frame = load_clean_signal_frame(record, common_signal_columns)
    time_seconds = (
        (frame["time"] - frame["time"].iloc[0]).dt.total_seconds().to_numpy(dtype=float)
    )
    signal = frame[args.column].to_numpy(dtype=float)
    sampling_rate = estimate_sampling_rate(time_seconds)
    frequencies, amplitudes = compute_single_sided_spectrum(signal, sampling_rate)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)

    axes[0].plot(time_seconds, signal, linewidth=0.8, color="#1f4e79")
    axes[0].set_title(f"Time Signal: Case {record.case_id} | {args.column}")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(frequencies, amplitudes, linewidth=1.0, color="#c84c09")
    axes[1].set_title(f"Single-Sided FFT Spectrum | Sampling Rate {sampling_rate:.2f} Hz")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Amplitude")
    axes[1].set_xlim(0.0, sampling_rate / 2.0)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(f"FFT Exploration | Case {record.case_id}")
    fig.savefig(args.output, dpi=180)
    plt.close(fig)

    dominant_index = int(np.argmax(amplitudes[1:]) + 1) if len(amplitudes) > 1 else 0
    dominant_frequency = float(frequencies[dominant_index]) if len(frequencies) else 0.0
    dominant_amplitude = float(amplitudes[dominant_index]) if len(amplitudes) else 0.0

    print(f"已输出图片: {args.output}")
    print(f"工况文件: {record.file_name}")
    print(f"分析通道: {args.column}")
    print(f"采样频率: {sampling_rate:.4f} Hz")
    print(f"主峰频率: {dominant_frequency:.4f} Hz")
    print(f"主峰幅值: {dominant_amplitude:.6f}")


if __name__ == "__main__":
    main()
