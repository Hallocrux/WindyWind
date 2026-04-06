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
WINDOW_SECONDS = 30.0
HALF_WINDOW_SECONDS = WINDOW_SECONDS / 2.0
OUTPUT_DIR = Path("outputs/try/007_case5_core_middle_30s_plots")
QUALITY_COLUMNS = {
    "__row_missing_count",
    "__row_has_missing",
    "__in_leading_missing_block",
    "__in_trailing_missing_block",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="绘制工况5更稳中段30秒的逐列图。")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    record, frame = load_case5_core_middle_frame()
    time_seconds = (frame["time"] - frame["time"].iloc[0]).dt.total_seconds().to_numpy(dtype=float)
    signal_columns = [c for c in frame.columns if c not in {"time", *QUALITY_COLUMNS}]
    acceleration_columns = [c for c in signal_columns if c.startswith("WSMS")]
    strain_columns = [c for c in signal_columns if c.startswith("应变传感器")]

    summary_rows = []
    for column in signal_columns:
        values = frame[column].to_numpy(dtype=float)
        summary_rows.append(
            {
                "column": column,
                "group": "acceleration" if column.startswith("WSMS") else "strain",
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "ptp": float(np.ptp(values)),
            }
        )
    summary_df = pd.DataFrame(summary_rows).sort_values(["group", "column"]).reset_index(drop=True)
    summary_df.to_csv(args.output_dir / "channel_summary.csv", index=False, encoding="utf-8-sig")

    create_small_multiples(
        frame=frame,
        columns=signal_columns,
        time_seconds=time_seconds,
        output_path=args.output_dir / "case5_core_middle_30s_small_multiples.png",
    )
    create_overlay(
        title="Case 5 core middle 30s | normalized acceleration channels",
        frame=frame,
        columns=acceleration_columns,
        time_seconds=time_seconds,
        output_path=args.output_dir / "case5_core_middle_30s_acceleration_overlay.png",
    )
    create_overlay(
        title="Case 5 core middle 30s | normalized strain channels",
        frame=frame,
        columns=strain_columns,
        time_seconds=time_seconds,
        output_path=args.output_dir / "case5_core_middle_30s_strain_overlay.png",
    )
    write_summary(
        record_name=record.file_name,
        frame=frame,
        acceleration_columns=acceleration_columns,
        strain_columns=strain_columns,
        summary_df=summary_df,
        output_path=args.output_dir / "summary.md",
    )


def load_case5_core_middle_frame():
    records = scan_dataset_records()
    record = next((item for item in records if item.case_id == CASE_ID), None)
    if record is None:
        raise ValueError("未找到工况5数据。")

    common_signal_columns = get_common_signal_columns(records)
    frame = load_clean_signal_frame(record, common_signal_columns)
    trimmed = frame[
        (frame["__in_leading_missing_block"] == 0)
        & (frame["__in_trailing_missing_block"] == 0)
    ].reset_index(drop=True)

    start = trimmed["time"].iloc[0]
    end = trimmed["time"].iloc[-1]
    center = start + (end - start) / 2
    half_window = pd.to_timedelta(HALF_WINDOW_SECONDS, unit="s")
    mid_start = center - half_window
    mid_end = center + half_window
    core = trimmed[(trimmed["time"] >= mid_start) & (trimmed["time"] <= mid_end)].reset_index(drop=True)
    if core.empty:
        raise ValueError("中心30秒截取后无数据。")
    return record, core


def create_small_multiples(
    frame: pd.DataFrame,
    columns: list[str],
    time_seconds: np.ndarray,
    output_path: Path,
) -> None:
    n_cols = 2
    n_rows = int(np.ceil(len(columns) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 2.5), constrained_layout=True)
    axes_flat = np.atleast_1d(axes).flatten()

    for ax, column in zip(axes_flat, columns):
        values = frame[column].to_numpy(dtype=float)
        color = "#1f4e79" if column.startswith("WSMS") else "#c84c09"
        ax.plot(time_seconds, values, color=color, linewidth=0.9)
        ax.set_title(display_name(column), fontsize=9)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.25)

    for ax in axes_flat[len(columns) :]:
        ax.axis("off")

    fig.suptitle("Case 5 core middle 30s | all effective channels", fontsize=14)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def create_overlay(
    title: str,
    frame: pd.DataFrame,
    columns: list[str],
    time_seconds: np.ndarray,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(14, 6), constrained_layout=True)
    for column in columns:
        values = frame[column].to_numpy(dtype=float)
        centered = values - float(np.mean(values))
        std = float(np.std(values))
        normalized = centered / std if std > 0 else centered
        ax.plot(time_seconds, normalized, linewidth=0.8, alpha=0.75, label=display_name(column))

    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Normalized value")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", ncol=2, fontsize=8)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_summary(
    record_name: str,
    frame: pd.DataFrame,
    acceleration_columns: list[str],
    strain_columns: list[str],
    summary_df: pd.DataFrame,
    output_path: Path,
) -> None:
    start = frame["time"].iloc[0]
    end = frame["time"].iloc[-1]
    duration = float((end - start).total_seconds())
    top_ptp = summary_df.sort_values("ptp", ascending=False).head(5)

    lines = [
        "# 工况5 更稳中段 30s 逐列图像",
        "",
        f"- 数据文件：`{record_name}`",
        f"- 当前时间范围：`{start}` 到 `{end}`",
        f"- 当前展示时长：`{duration:.3f}s`",
        f"- 有效信号列数：`{len(acceleration_columns) + len(strain_columns)}`",
        f"- 加速度通道数：`{len(acceleration_columns)}`",
        f"- 应变通道数：`{len(strain_columns)}`",
        "",
        "## 幅值范围最大的列",
        "",
    ]
    for _, row in top_ptp.iterrows():
        lines.append(f"- `{row['column']}`: `ptp={row['ptp']:.3f}`, `std={row['std']:.3f}`")

    lines.extend(
        [
            "",
            "## 说明",
            "",
            "- 当前图像展示的是工况5稳定保留段中心30秒。",
            "- 加速度与应变总览图均做了按列标准差归一化，便于比较波形节奏。",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def display_name(column: str) -> str:
    if column.startswith("应变传感器") and column.endswith(".chdata"):
        index = column.replace("应变传感器", "").replace(".chdata", "")
        return f"SG{index}.chdata"
    return column


if __name__ == "__main__":
    main()
