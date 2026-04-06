from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.current.data_loading import (
    get_common_signal_columns,
    load_clean_signal_frame,
    scan_dataset_records,
)
from src.current.features import WindowConfig

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_OUTPUT_DIR = Path("outputs/try/018_structural_fundamental_frequency_scan")


@dataclass(frozen=True)
class ScanConfig:
    sampling_rate: float = 50.0
    window_size: int = 250
    step_size: int = 125
    freq_min: float = 0.5
    freq_max: float = 8.0
    rotor_exclusion_width: float = 0.2
    max_rotor_harmonic: int = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="扫描结构基频候选并输出汇总结果。")
    parser.add_argument(
        "--freq-min", type=float, default=0.5, help="候选频率搜索下限（Hz）。"
    )
    parser.add_argument(
        "--freq-max", type=float, default=8.0, help="候选频率搜索上限（Hz）。"
    )
    parser.add_argument(
        "--rotor-exclusion-width",
        type=float,
        default=0.2,
        help="转频及其倍频屏蔽半宽（Hz）。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="输出目录。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ScanConfig(
        freq_min=args.freq_min,
        freq_max=args.freq_max,
        rotor_exclusion_width=args.rotor_exclusion_width,
    )

    records = scan_dataset_records()
    common_signal_columns = get_common_signal_columns(records)
    strain_columns = [
        column for column in common_signal_columns if column.startswith("应变传感器")
    ]
    acc_columns = [column for column in common_signal_columns if ".Acc" in column]
    if not strain_columns:
        raise ValueError("未找到可用于结构基频扫描的应变通道。")
    if not acc_columns:
        raise ValueError("未找到可用于结构基频扫描的加速度通道。")

    window_config = WindowConfig(
        sampling_rate=config.sampling_rate,
        window_size=config.window_size,
        step_size=config.step_size,
    )
    window_rows: list[dict[str, float | int | str | None]] = []
    case_rows: list[dict[str, float | int | None]] = []

    for record in records:
        frame = load_clean_signal_frame(record, common_signal_columns)
        rotor_hz = record.rpm / 60.0 if record.rpm is not None else None
        case_window_rows = collect_case_window_rows(
            record=record,
            frame=frame,
            strain_columns=strain_columns,
            acc_columns=acc_columns,
            window_config=window_config,
            config=config,
            rotor_hz=rotor_hz,
        )
        window_rows.extend(case_window_rows)
        case_rows.append(
            build_case_summary(record.case_id, record.rpm, rotor_hz, case_window_rows)
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    window_df = pd.DataFrame(window_rows)
    case_df = pd.DataFrame(case_rows).sort_values("case_id").reset_index(drop=True)

    window_df.to_csv(
        args.output_dir / "window_frequency_candidates.csv",
        index=False,
        encoding="utf-8-sig",
    )
    case_df.to_csv(
        args.output_dir / "case_frequency_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    save_overview_figure(
        case_df, window_df, args.output_dir / "structural_frequency_overview.png"
    )

    print(f"已输出目录: {args.output_dir}")
    print("工况级候选频率摘要:")
    print(
        case_df[
            [
                "case_id",
                "rotor_hz",
                "strain_candidate_freq_median",
                "acc_candidate_freq_median",
                "strain_window_count",
                "acc_window_count",
            ]
        ].to_string(index=False)
    )


def collect_case_window_rows(
    *,
    record,
    frame: pd.DataFrame,
    strain_columns: list[str],
    acc_columns: list[str],
    window_config: WindowConfig,
    config: ScanConfig,
    rotor_hz: float | None,
) -> list[dict[str, float | int | str | None]]:
    rows: list[dict[str, float | int | str | None]] = []
    window_index = 0
    for _, segment_df in frame.groupby("__segment_id", sort=True):
        segment_df = segment_df.reset_index(drop=True)
        if len(segment_df) < window_config.window_size:
            continue

        for start in range(
            0, len(segment_df) - window_config.window_size + 1, window_config.step_size
        ):
            window = segment_df.iloc[start : start + window_config.window_size]
            shared_payload = {
                "case_id": int(record.case_id),
                "window_index": int(window_index),
                "start_time": window["time"].iloc[0],
                "end_time": window["time"].iloc[-1],
                "rpm": record.rpm,
                "rotor_hz": rotor_hz,
            }
            strain_freqs = [
                extract_candidate_frequency(
                    signal=window[column].to_numpy(dtype=float, copy=False),
                    sampling_rate=config.sampling_rate,
                    freq_min=config.freq_min,
                    freq_max=config.freq_max,
                    rotor_hz=rotor_hz,
                    rotor_exclusion_width=config.rotor_exclusion_width,
                    max_rotor_harmonic=config.max_rotor_harmonic,
                )
                for column in strain_columns
            ]
            acc_freqs = [
                extract_candidate_frequency(
                    signal=window[column].to_numpy(dtype=float, copy=False),
                    sampling_rate=config.sampling_rate,
                    freq_min=config.freq_min,
                    freq_max=config.freq_max,
                    rotor_hz=rotor_hz,
                    rotor_exclusion_width=config.rotor_exclusion_width,
                    max_rotor_harmonic=config.max_rotor_harmonic,
                )
                for column in acc_columns
            ]

            rows.append(
                {
                    **shared_payload,
                    "sensor_group": "strain",
                    "candidate_freq": robust_median(strain_freqs),
                    "candidate_freq_std": robust_std(strain_freqs),
                    "channel_count": count_finite(strain_freqs),
                }
            )
            rows.append(
                {
                    **shared_payload,
                    "sensor_group": "acc",
                    "candidate_freq": robust_median(acc_freqs),
                    "candidate_freq_std": robust_std(acc_freqs),
                    "channel_count": count_finite(acc_freqs),
                }
            )
            window_index += 1
    return rows


def extract_candidate_frequency(
    *,
    signal: np.ndarray,
    sampling_rate: float,
    freq_min: float,
    freq_max: float,
    rotor_hz: float | None,
    rotor_exclusion_width: float,
    max_rotor_harmonic: int,
) -> float:
    centered = np.asarray(signal, dtype=float) - float(np.mean(signal))
    window = np.hanning(centered.size)
    spectrum = np.fft.rfft(centered * window)
    magnitudes = np.abs(spectrum)
    freqs = np.fft.rfftfreq(centered.size, d=1.0 / sampling_rate)
    if magnitudes.size == 0:
        return np.nan

    valid_mask = (freqs >= freq_min) & (freqs <= freq_max)
    if rotor_hz is not None and rotor_hz > 0:
        for harmonic in range(1, max_rotor_harmonic + 1):
            valid_mask &= np.abs(freqs - rotor_hz * harmonic) > rotor_exclusion_width

    if not np.any(valid_mask):
        return np.nan

    masked = np.where(valid_mask, magnitudes, -1.0)
    peak_index = int(np.argmax(masked))
    if masked[peak_index] < 0:
        return np.nan
    return float(freqs[peak_index])


def build_case_summary(
    case_id: int,
    rpm: float | None,
    rotor_hz: float | None,
    case_window_rows: list[dict[str, float | int | str | None]],
) -> dict[str, float | int | None]:
    case_df = pd.DataFrame(case_window_rows)
    strain_values = case_df.loc[
        case_df["sensor_group"] == "strain", "candidate_freq"
    ].to_numpy(dtype=float)
    acc_values = case_df.loc[
        case_df["sensor_group"] == "acc", "candidate_freq"
    ].to_numpy(dtype=float)
    return {
        "case_id": int(case_id),
        "rpm": rpm,
        "rotor_hz": rotor_hz,
        "strain_candidate_freq_median": robust_median(strain_values),
        "strain_candidate_freq_iqr": robust_iqr(strain_values),
        "strain_window_count": count_finite(strain_values),
        "acc_candidate_freq_median": robust_median(acc_values),
        "acc_candidate_freq_iqr": robust_iqr(acc_values),
        "acc_window_count": count_finite(acc_values),
    }


def save_overview_figure(
    case_df: pd.DataFrame, window_df: pd.DataFrame, output_path: Path
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)

    x = case_df["case_id"].to_numpy(dtype=int)
    axes[0].plot(
        x,
        case_df["strain_candidate_freq_median"],
        marker="o",
        linewidth=1.5,
        label="strain candidate",
    )
    axes[0].plot(
        x,
        case_df["acc_candidate_freq_median"],
        marker="s",
        linewidth=1.2,
        label="acc candidate",
    )
    if case_df["rotor_hz"].notna().any():
        axes[0].plot(
            x,
            case_df["rotor_hz"],
            marker="^",
            linewidth=1.0,
            linestyle="--",
            label="rotor frequency",
        )
    axes[0].set_title("Case-Level Candidate Frequency vs Rotor Frequency")
    axes[0].set_xlabel("Case ID")
    axes[0].set_ylabel("Frequency (Hz)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    strain_window_df = window_df.loc[window_df["sensor_group"] == "strain"].copy()
    strain_values = strain_window_df["candidate_freq"].dropna().to_numpy(dtype=float)
    if strain_values.size:
        bins = np.arange(0.5, 8.25, 0.2)
        axes[1].hist(
            strain_values, bins=bins, color="#c84c09", alpha=0.85, edgecolor="white"
        )
    axes[1].set_title("Strain-Group Candidate Frequency Distribution")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Window Count")
    axes[1].grid(True, alpha=0.3)

    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def robust_median(values: list[float] | np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return np.nan
    return float(np.median(array))


def robust_std(values: list[float] | np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return np.nan
    return float(np.std(array, ddof=0))


def robust_iqr(values: list[float] | np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return np.nan
    return float(np.quantile(array, 0.75) - np.quantile(array, 0.25))


def count_finite(values: list[float] | np.ndarray) -> int:
    array = np.asarray(values, dtype=float)
    return int(np.isfinite(array).sum())


if __name__ == "__main__":
    main()
