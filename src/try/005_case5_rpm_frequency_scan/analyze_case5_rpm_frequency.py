from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.current.data_loading import get_common_signal_columns, load_clean_signal_frame, scan_dataset_records  # noqa: E402

CASE_ID = 5
RPM = 166.0
OUTPUT_DIR = Path("outputs/try/005_case5_rpm_frequency_scan")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="绘制工况5稳定保留段的 RPM 频域图。")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    record, frame = load_case5_trimmed_frame()
    time_seconds = (frame["time"] - frame["time"].iloc[0]).dt.total_seconds().to_numpy(dtype=float)
    sampling_rate = 1.0 / float(np.median(np.diff(time_seconds)))
    accel_columns = [column for column in frame.columns if column.startswith("WSMS")]

    target_rows: list[dict[str, object]] = []
    spectra: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    selected_scores: list[tuple[str, float]] = []
    rpm_hz = RPM / 60.0
    markers = [
        ("1x rpm", rpm_hz),
        ("2x rpm", rpm_hz * 2.0),
        ("3x rpm", rpm_hz * 3.0),
        ("4x rpm", rpm_hz * 4.0),
    ]

    for column in accel_columns:
        signal = frame[column].to_numpy(dtype=float)
        freqs, amps = compute_spectrum(signal, sampling_rate)
        spectra[column] = (freqs, amps)
        mask = (freqs >= 0.5) & (freqs <= 12.0)
        band_freqs = freqs[mask]
        band_amps = amps[mask]
        top_idx = int(np.argmax(band_amps))
        top_amp = float(band_amps[top_idx])
        score = 0.0

        for label, target_hz in markers:
            idx = int(np.argmin(np.abs(freqs - target_hz)))
            amp = float(amps[idx])
            ratio = amp / top_amp if top_amp > 0 else np.nan
            target_rows.append(
                {
                    "column": column,
                    "marker": label,
                    "target_hz": target_hz,
                    "nearest_hz": float(freqs[idx]),
                    "amplitude": amp,
                    "ratio_to_top_0_5_12hz": ratio,
                    "top_peak_hz_0_5_12hz": float(band_freqs[top_idx]),
                    "top_peak_amp_0_5_12hz": top_amp,
                }
            )
            if label in {"1x rpm", "3x rpm"} and np.isfinite(ratio):
                score += ratio

        selected_scores.append((column, score))

    target_df = pd.DataFrame(target_rows)
    target_df.to_csv(args.output_dir / "case5_frequency_targets.csv", index=False, encoding="utf-8-sig")

    selected_columns = [name for name, _ in sorted(selected_scores, key=lambda item: item[1], reverse=True)[:4]]
    create_overview_plot(
        record_name=record.file_name,
        spectra=spectra,
        markers=markers,
        output_path=args.output_dir / "case5_trimmed_acc_overview.png",
    )
    create_selected_plot(
        record_name=record.file_name,
        spectra=spectra,
        selected_columns=selected_columns,
        markers=markers,
        output_path=args.output_dir / "case5_trimmed_selected_channels.png",
    )
    write_summary(
        target_df=target_df,
        selected_columns=selected_columns,
        output_path=args.output_dir / "summary.md",
    )


def load_case5_trimmed_frame():
    records = scan_dataset_records()
    record = next((item for item in records if item.case_id == CASE_ID), None)
    if record is None:
        raise ValueError("未找到工况5数据。")

    common_signal_columns = get_common_signal_columns(records)
    frame = load_clean_signal_frame(record, common_signal_columns)
    frame = frame[
        (frame["__in_leading_missing_block"] == 0)
        & (frame["__in_trailing_missing_block"] == 0)
    ].reset_index(drop=True)
    return record, frame


def compute_spectrum(signal: np.ndarray, sampling_rate: float) -> tuple[np.ndarray, np.ndarray]:
    centered = signal - np.mean(signal)
    window = np.hanning(len(centered))
    windowed = centered * window
    fft_values = np.fft.rfft(windowed)
    frequencies = np.fft.rfftfreq(len(centered), d=1.0 / sampling_rate)
    scale = 2.0 / np.sum(window)
    amplitudes = scale * np.abs(fft_values)
    if len(amplitudes) > 0:
        amplitudes[0] *= 0.5
    return frequencies, amplitudes


def create_overview_plot(
    record_name: str,
    spectra: dict[str, tuple[np.ndarray, np.ndarray]],
    markers: list[tuple[str, float]],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), constrained_layout=True)

    normalized_curves = []
    for column, (freqs, amps) in spectra.items():
        mask = (freqs >= 0.5) & (freqs <= 12.0)
        plot_freqs = freqs[mask]
        plot_amps = amps[mask]
        peak = float(np.max(plot_amps)) if len(plot_amps) else 1.0
        normalized = plot_amps / peak if peak > 0 else plot_amps
        normalized_curves.append(normalized)
        axes[0].plot(plot_freqs, normalized, alpha=0.28, linewidth=1.0, label=column)

    mean_curve = np.mean(np.vstack(normalized_curves), axis=0)
    axes[0].plot(plot_freqs, mean_curve, color="black", linewidth=2.0, label="mean normalized")
    axes[0].set_title("Case 5 trimmed segment | normalized spectra")
    axes[0].set_xlabel("Frequency (Hz)")
    axes[0].set_ylabel("Normalized amplitude")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(0.5, 12.0)

    mean_abs_curve = np.mean(
        np.vstack([amps[(freqs >= 0.5) & (freqs <= 12.0)] for freqs, amps in spectra.values()]),
        axis=0,
    )
    axes[1].plot(plot_freqs, mean_abs_curve, color="#1f4e79", linewidth=2.0)
    axes[1].set_title("Mean amplitude spectrum across acceleration channels")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Amplitude")
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(0.5, 12.0)

    for ax in axes:
        for label, target_hz in markers:
            ax.axvline(target_hz, color="#c84c09", linestyle="--", linewidth=1.0, alpha=0.8)
            ax.text(target_hz + 0.03, ax.get_ylim()[1] * 0.9, label, rotation=90, color="#c84c09", fontsize=9)

    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def create_selected_plot(
    record_name: str,
    spectra: dict[str, tuple[np.ndarray, np.ndarray]],
    selected_columns: list[str],
    markers: list[tuple[str, float]],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    axes_flat = axes.flatten()
    for ax, column in zip(axes_flat, selected_columns):
        freqs, amps = spectra[column]
        mask = (freqs >= 0.5) & (freqs <= 12.0)
        plot_freqs = freqs[mask]
        plot_amps = amps[mask]
        ax.plot(plot_freqs, plot_amps, color="#1f4e79", linewidth=1.2)
        for label, target_hz in markers:
            ax.axvline(target_hz, color="#c84c09", linestyle="--", linewidth=0.9, alpha=0.8)
        ax.set_title(column)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Amplitude")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0.5, 12.0)

    for ax in axes_flat[len(selected_columns) :]:
        ax.axis("off")

    fig.suptitle("Case 5 selected channels")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_summary(target_df: pd.DataFrame, selected_columns: list[str], output_path: Path) -> None:
    pivot = (
        target_df.pivot(index="column", columns="marker", values="ratio_to_top_0_5_12hz")
        .sort_values(["3x rpm", "1x rpm"], ascending=False)
    )
    top_1x = pivot["1x rpm"].sort_values(ascending=False).head(5)
    top_3x = pivot["3x rpm"].sort_values(ascending=False).head(5)
    strong_1x = int((pivot["1x rpm"] >= 0.5).sum())
    strong_3x = int((pivot["3x rpm"] >= 0.5).sum())

    lines = [
        "# 工况5 RPM 频域诊断结论",
        "",
        "- 当前分析对象是删除首尾连续缺失段后的稳定保留段。",
        f"- 目标转速：`166 rpm`，对应 `1x rpm = {RPM / 60.0:.4f} Hz`。",
        f"- 当前重点通道：`{', '.join(selected_columns)}`",
        "",
        "## 观察",
        "",
        f"- `1x rpm` 相对通道主峰比例 >= 0.5 的加速度通道数：`{strong_1x}`",
        f"- `3x rpm` 相对通道主峰比例 >= 0.5 的加速度通道数：`{strong_3x}`",
        "- 当前更像是：`1x rpm` 本身并不普遍成为最强峰，但 `3x rpm` 在部分通道里更接近显著成分。",
        "",
        "## 1x rpm 最明显的通道",
        "",
    ]
    for column, value in top_1x.items():
        lines.append(f"- `{column}`: 相对主峰比例 `{value:.3f}`")

    lines.extend(
        [
            "",
            "## 3x rpm 最明显的通道",
            "",
        ]
    )
    for column, value in top_3x.items():
        lines.append(f"- `{column}`: 相对主峰比例 `{value:.3f}`")

    lines.extend(
        [
            "",
            "## 当前判断",
            "",
            "- 目前频域图不能简单说“166 rpm 在 2.767 Hz 上形成了压倒性主峰”。",
            "- 但如果按三叶片转子的倍频理解，`3x rpm = 8.30 Hz` 在部分加速度通道上更容易被看到。",
        ]
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
