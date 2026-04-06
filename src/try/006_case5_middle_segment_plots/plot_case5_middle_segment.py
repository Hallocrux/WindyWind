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
OUTPUT_DIR = Path("outputs/try/006_case5_middle_segment_plots")
QUALITY_COLUMNS = {
    "__row_missing_count",
    "__row_has_missing",
    "__in_leading_missing_block",
    "__in_trailing_missing_block",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="绘制工况5中间非缺失部分的逐列图。")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    record, frame = load_case5_trimmed_frame()
    signal_columns = [c for c in frame.columns if c not in {"time", *QUALITY_COLUMNS}]
    acceleration_columns = [c for c in signal_columns if c.startswith("WSMS")]
    strain_columns = [c for c in signal_columns if c.startswith("应变传感器")]
    time_seconds = (frame["time"] - frame["time"].iloc[0]).dt.total_seconds().to_numpy(dtype=float)

    summary_rows = []
    for column in signal_columns:
        series = frame[column].to_numpy(dtype=float)
        summary_rows.append(
            {
                "column": column,
                "group": "acceleration" if column.startswith("WSMS") else "strain",
                "mean": float(np.mean(series)),
                "std": float(np.std(series)),
                "min": float(np.min(series)),
                "max": float(np.max(series)),
                "ptp": float(np.ptp(series)),
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values(["group", "column"]).reset_index(drop=True)
    summary_df.to_csv(args.output_dir / "channel_summary.csv", index=False, encoding="utf-8-sig")

    create_small_multiples(
        record_name=record.file_name,
        frame=frame,
        columns=signal_columns,
        time_seconds=time_seconds,
        output_path=args.output_dir / "case5_trimmed_all_channels_small_multiples.png",
    )
    create_overlay(
        title="Case 5 trimmed segment | normalized acceleration channels",
        frame=frame,
        columns=acceleration_columns,
        time_seconds=time_seconds,
        output_path=args.output_dir / "case5_trimmed_acceleration_overlay.png",
    )
    create_overlay(
        title="Case 5 trimmed segment | normalized strain channels",
        frame=frame,
        columns=strain_columns,
        time_seconds=time_seconds,
        output_path=args.output_dir / "case5_trimmed_strain_overlay.png",
    )
    write_summary(
        record_name=record.file_name,
        frame=frame,
        acceleration_columns=acceleration_columns,
        strain_columns=strain_columns,
        summary_df=summary_df,
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


def create_small_multiples(
    record_name: str,
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
        series = frame[column].to_numpy(dtype=float)
        color = "#1f4e79" if column.startswith("WSMS") else "#c84c09"
        ax.plot(time_seconds, series, color=color, linewidth=0.8)
        ax.set_title(display_name(column), fontsize=9)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.25)

    for ax in axes_flat[len(columns) :]:
        ax.axis("off")

    fig.suptitle("Case 5 trimmed segment | all effective channels", fontsize=14)
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
        std = float(np.std(values))
        normalized = values - float(np.mean(values))
        if std > 0:
            normalized = normalized / std
        ax.plot(time_seconds, normalized, linewidth=0.8, alpha=0.7, label=display_name(column))

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
    duration = float((frame["time"].iloc[-1] - frame["time"].iloc[0]).total_seconds())
    top_ptp = summary_df.sort_values("ptp", ascending=False).head(5)

    lines = [
        "# 工况5 中间非缺失部分逐列图像",
        "",
        f"- 数据文件：`{record_name}`",
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
            "- 小图输出的是删除首尾连续缺失段后的整段保留数据。",
            "- 加速度与应变总览图做了按列标准差归一化，便于比较不同列的波形节奏而不是绝对量纲。",
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
