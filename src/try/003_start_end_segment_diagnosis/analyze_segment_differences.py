from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.current.data_loading import (
    DatasetRecord,
    get_common_signal_columns,
    load_clean_signal_frame,
    scan_dataset_records,
)
from src.current.features import WindowConfig, build_case_feature_frame

SEGMENT_LABELS = ("start", "middle", "end")
DEFAULT_OUTPUT_DIR = Path("outputs/try/003_start_end_segment_diagnosis")


@dataclass(frozen=True)
class SegmentBounds:
    label: str
    start_sec: float
    end_sec: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="比较每个工况开始/中段/结束 15s 的窗口差异。")
    parser.add_argument(
        "--segment-seconds",
        type=float,
        default=15.0,
        help="每个比较时间段的持续时间，单位秒。",
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
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    records = scan_dataset_records()
    common_columns = get_common_signal_columns(records)
    window_config = WindowConfig()

    segment_rows: list[dict[str, object]] = []
    distance_rows: list[dict[str, object]] = []

    for record in records:
        cleaned = load_clean_signal_frame(record, common_columns)
        feature_df = build_case_feature_frame(record, cleaned, window_config)
        feature_df = add_relative_time_columns(feature_df, cleaned)
        bounds = build_segment_bounds(cleaned, args.segment_seconds)
        feature_df["segment"] = assign_segments(feature_df, bounds)
        segment_df = feature_df[feature_df["segment"].notna()].copy()

        if segment_df.empty:
            continue

        segment_rows.extend(
            summarize_case_segments(record, segment_df, args.segment_seconds).to_dict("records")
        )
        distance_rows.extend(
            summarize_case_distances(record, segment_df).to_dict("records")
        )

    segment_summary = pd.DataFrame(segment_rows).sort_values(["case_id", "segment"]).reset_index(drop=True)
    distance_summary = pd.DataFrame(distance_rows).sort_values("case_id").reset_index(drop=True)
    overall_summary = build_overall_summary(segment_summary, distance_summary)

    segment_summary.to_csv(output_dir / "segment_window_summary.csv", index=False, encoding="utf-8-sig")
    distance_summary.to_csv(output_dir / "segment_distance_summary.csv", index=False, encoding="utf-8-sig")
    overall_summary.to_csv(output_dir / "segment_overall_summary.csv", index=False, encoding="utf-8-sig")

    create_overview_plot(segment_summary, distance_summary, output_dir / "segment_differences.png")
    write_markdown_summary(
        segment_summary=segment_summary,
        distance_summary=distance_summary,
        overall_summary=overall_summary,
        output_path=output_dir / "summary.md",
        segment_seconds=args.segment_seconds,
    )


def add_relative_time_columns(feature_df: pd.DataFrame, cleaned_signal: pd.DataFrame) -> pd.DataFrame:
    origin = cleaned_signal["time"].iloc[0]
    feature_df = feature_df.copy()
    feature_df["start_sec"] = (feature_df["start_time"] - origin).dt.total_seconds()
    feature_df["end_sec"] = (feature_df["end_time"] - origin).dt.total_seconds()
    feature_df["center_sec"] = (feature_df["start_sec"] + feature_df["end_sec"]) / 2.0
    return feature_df


def build_segment_bounds(cleaned_signal: pd.DataFrame, segment_seconds: float) -> list[SegmentBounds]:
    duration = float((cleaned_signal["time"].iloc[-1] - cleaned_signal["time"].iloc[0]).total_seconds())
    if duration < segment_seconds:
        raise ValueError(f"工况时长不足 {segment_seconds} 秒，无法分段比较。")

    middle_start = max(0.0, duration / 2.0 - segment_seconds / 2.0)
    middle_end = min(duration, middle_start + segment_seconds)
    if middle_end - middle_start < segment_seconds:
        middle_start = max(0.0, duration - segment_seconds)
        middle_end = duration

    end_start = max(0.0, duration - segment_seconds)
    return [
        SegmentBounds("start", 0.0, segment_seconds),
        SegmentBounds("middle", middle_start, middle_end),
        SegmentBounds("end", end_start, duration),
    ]


def assign_segments(feature_df: pd.DataFrame, bounds: list[SegmentBounds]) -> pd.Series:
    labels: list[str | None] = []
    for _, row in feature_df.iterrows():
        label = None
        for segment in bounds:
            if row["start_sec"] >= segment.start_sec and row["end_sec"] <= segment.end_sec:
                label = segment.label
                break
        labels.append(label)
    return pd.Series(labels, index=feature_df.index, dtype="object")


def summarize_case_segments(
    record: DatasetRecord,
    segment_df: pd.DataFrame,
    segment_seconds: float,
) -> pd.DataFrame:
    rms_columns = [column for column in segment_df.columns if column.endswith("__rms")]
    peak_freq_columns = [column for column in segment_df.columns if column.endswith("__fft_peak_freq")]
    energy_columns = [column for column in segment_df.columns if column.endswith("__fft_total_energy")]

    rows: list[dict[str, object]] = []
    for label in SEGMENT_LABELS:
        block = segment_df[segment_df["segment"] == label].copy()
        rows.append(
            {
                "case_id": record.case_id,
                "file_name": record.file_name,
                "segment": label,
                "segment_seconds": segment_seconds,
                "window_count": int(len(block)),
                "avg_raw_missing_ratio": float(block["raw_missing_ratio"].mean()) if not block.empty else np.nan,
                "dirty_window_ratio": float((block["raw_missing_ratio"] > 0).mean()) if not block.empty else np.nan,
                "leading_touch_ratio": float((block["touches_leading_missing"] > 0).mean()) if not block.empty else np.nan,
                "trailing_touch_ratio": float((block["touches_trailing_missing"] > 0).mean()) if not block.empty else np.nan,
                "avg_rms": float(block[rms_columns].mean(axis=1).mean()) if not block.empty else np.nan,
                "avg_fft_peak_freq": float(block[peak_freq_columns].mean(axis=1).mean()) if not block.empty else np.nan,
                "avg_fft_total_energy": float(block[energy_columns].mean(axis=1).mean()) if not block.empty else np.nan,
                "median_center_sec": float(block["center_sec"].median()) if not block.empty else np.nan,
            }
        )
    return pd.DataFrame(rows)


def summarize_case_distances(record: DatasetRecord, segment_df: pd.DataFrame) -> pd.DataFrame:
    vibration_columns = [
        column
        for column in segment_df.columns
        if "__" in column and not column.startswith("__")
    ]
    scaled = StandardScaler().fit_transform(segment_df[vibration_columns])
    scaled_df = pd.DataFrame(scaled, columns=vibration_columns, index=segment_df.index)
    scaled_df["segment"] = segment_df["segment"].to_numpy()

    centroid_map: dict[str, np.ndarray] = {}
    dispersion_map: dict[str, float] = {}
    count_map: dict[str, int] = {}
    for label in SEGMENT_LABELS:
        block = scaled_df[scaled_df["segment"] == label].drop(columns=["segment"])
        count_map[label] = int(len(block))
        if block.empty:
            centroid_map[label] = np.full(len(vibration_columns), np.nan)
            dispersion_map[label] = np.nan
            continue
        centroid = block.mean(axis=0).to_numpy(dtype=float)
        centroid_map[label] = centroid
        dispersion_map[label] = float(
            np.linalg.norm(block.to_numpy(dtype=float) - centroid, axis=1).mean()
        )

    start_mid = safe_distance(centroid_map["start"], centroid_map["middle"])
    end_mid = safe_distance(centroid_map["end"], centroid_map["middle"])
    start_end = safe_distance(centroid_map["start"], centroid_map["end"])

    return pd.DataFrame(
        [
            {
                "case_id": record.case_id,
                "file_name": record.file_name,
                "start_window_count": count_map["start"],
                "middle_window_count": count_map["middle"],
                "end_window_count": count_map["end"],
                "start_middle_distance": start_mid,
                "end_middle_distance": end_mid,
                "start_end_distance": start_end,
                "start_dispersion": dispersion_map["start"],
                "middle_dispersion": dispersion_map["middle"],
                "end_dispersion": dispersion_map["end"],
                "start_middle_vs_within": safe_ratio(
                    start_mid,
                    np.nanmean([dispersion_map["start"], dispersion_map["middle"]]),
                ),
                "end_middle_vs_within": safe_ratio(
                    end_mid,
                    np.nanmean([dispersion_map["end"], dispersion_map["middle"]]),
                ),
                "start_end_vs_within": safe_ratio(
                    start_end,
                    np.nanmean([dispersion_map["start"], dispersion_map["end"]]),
                ),
            }
        ]
    )


def safe_distance(left: np.ndarray, right: np.ndarray) -> float:
    if np.isnan(left).any() or np.isnan(right).any():
        return np.nan
    return float(np.linalg.norm(left - right))


def safe_ratio(value: float, base: float) -> float:
    if np.isnan(value) or np.isnan(base) or base == 0.0:
        return np.nan
    return float(value / base)


def build_overall_summary(
    segment_summary: pd.DataFrame,
    distance_summary: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for label in SEGMENT_LABELS:
        block = segment_summary[segment_summary["segment"] == label]
        rows.append(
            {
                "metric_group": "segment_quality",
                "name": label,
                "window_count_mean": float(block["window_count"].mean()),
                "dirty_window_ratio_mean": float(block["dirty_window_ratio"].mean()),
                "avg_raw_missing_ratio_mean": float(block["avg_raw_missing_ratio"].mean()),
                "leading_touch_ratio_mean": float(block["leading_touch_ratio"].mean()),
                "trailing_touch_ratio_mean": float(block["trailing_touch_ratio"].mean()),
                "avg_rms_mean": float(block["avg_rms"].mean()),
                "avg_fft_peak_freq_mean": float(block["avg_fft_peak_freq"].mean()),
                "avg_fft_total_energy_mean": float(block["avg_fft_total_energy"].mean()),
            }
        )

    rows.extend(
        [
            {
                "metric_group": "distance",
                "name": "start_vs_middle",
                "window_count_mean": np.nan,
                "dirty_window_ratio_mean": np.nan,
                "avg_raw_missing_ratio_mean": np.nan,
                "leading_touch_ratio_mean": np.nan,
                "trailing_touch_ratio_mean": np.nan,
                "avg_rms_mean": float(distance_summary["start_middle_distance"].median()),
                "avg_fft_peak_freq_mean": float(distance_summary["start_middle_vs_within"].median()),
                "avg_fft_total_energy_mean": float(
                    (distance_summary["start_middle_vs_within"] > 1.0).mean()
                ),
            },
            {
                "metric_group": "distance",
                "name": "end_vs_middle",
                "window_count_mean": np.nan,
                "dirty_window_ratio_mean": np.nan,
                "avg_raw_missing_ratio_mean": np.nan,
                "leading_touch_ratio_mean": np.nan,
                "trailing_touch_ratio_mean": np.nan,
                "avg_rms_mean": float(distance_summary["end_middle_distance"].median()),
                "avg_fft_peak_freq_mean": float(distance_summary["end_middle_vs_within"].median()),
                "avg_fft_total_energy_mean": float(
                    (distance_summary["end_middle_vs_within"] > 1.0).mean()
                ),
            },
        ]
    )

    return pd.DataFrame(rows)


def create_overview_plot(
    segment_summary: pd.DataFrame,
    distance_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    quality_stats = (
        segment_summary.groupby("segment", sort=False)[
            ["dirty_window_ratio", "leading_touch_ratio", "trailing_touch_ratio"]
        ]
        .mean()
        .reindex(SEGMENT_LABELS)
    )
    quality_stats.plot(kind="bar", ax=axes[0])
    axes[0].set_title("Dirty-window and edge-missing ratios")
    axes[0].set_ylabel("ratio")
    axes[0].tick_params(axis="x", rotation=0)

    distance_plot = distance_summary.set_index("case_id")[
        ["start_middle_vs_within", "end_middle_vs_within"]
    ]
    distance_plot.plot(kind="bar", ax=axes[1])
    axes[1].axhline(1.0, color="black", linestyle="--", linewidth=1)
    axes[1].set_title("Boundary vs middle feature separation")
    axes[1].set_ylabel("distance / within-segment dispersion")

    energy_stats = (
        segment_summary.groupby("segment", sort=False)[["avg_rms", "avg_fft_total_energy"]]
        .mean()
        .reindex(SEGMENT_LABELS)
    )
    energy_stats.plot(kind="bar", ax=axes[2])
    axes[2].set_title("Mean activity levels by segment")
    axes[2].set_ylabel("mean value")
    axes[2].tick_params(axis="x", rotation=0)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_markdown_summary(
    segment_summary: pd.DataFrame,
    distance_summary: pd.DataFrame,
    overall_summary: pd.DataFrame,
    output_path: Path,
    segment_seconds: float,
) -> None:
    quality_stats = (
        segment_summary.groupby("segment", sort=False)[
            ["dirty_window_ratio", "leading_touch_ratio", "trailing_touch_ratio"]
        ]
        .mean()
        .reindex(SEGMENT_LABELS)
    )
    distance_start = distance_summary["start_middle_vs_within"]
    distance_end = distance_summary["end_middle_vs_within"]
    start_case_count = int((distance_start > 1.0).sum())
    end_case_count = int((distance_end > 1.0).sum())
    total_case_count = int(len(distance_summary))

    likely_drop_start = (distance_start > 1.0).mean() >= 0.5 or quality_stats.loc["start", "dirty_window_ratio"] > 0.2
    likely_drop_end = (distance_end > 1.0).mean() >= 0.5 or quality_stats.loc["end", "dirty_window_ratio"] > 0.2

    lines = [
        "# 开始/中段/结束 15s 差异诊断结论",
        "",
        f"- 分段时长：`{segment_seconds:.1f}s`",
        f"- 工况数：`{segment_summary['case_id'].nunique()}`",
        "",
        "## 整体观察",
        "",
        f"- 开始段平均脏窗口占比：`{quality_stats.loc['start', 'dirty_window_ratio']:.2%}`",
        f"- 中段平均脏窗口占比：`{quality_stats.loc['middle', 'dirty_window_ratio']:.2%}`",
        f"- 结束段平均脏窗口占比：`{quality_stats.loc['end', 'dirty_window_ratio']:.2%}`",
        f"- `start vs middle` 的相对分离度中位数：`{distance_start.median():.3f}`",
        f"- `end vs middle` 的相对分离度中位数：`{distance_end.median():.3f}`",
        f"- `start vs middle` 明显偏离（> 1.0）的工况数：`{start_case_count}/{total_case_count}`",
        f"- `end vs middle` 明显偏离（> 1.0）的工况数：`{end_case_count}/{total_case_count}`",
        "",
        "## 训练口径建议",
        "",
        f"- 开始段：{'建议优先排除或单独建模' if likely_drop_start else '暂不支持整体排除，需更细分到个别工况'}",
        f"- 结束段：{'建议优先排除或单独建模' if likely_drop_end else '暂不支持整体排除，需更细分到个别工况'}",
        "",
        "## 说明",
        "",
        "- 相对分离度大于 `1.0` 表示段间中心差异已经超过了典型段内波动量级。",
        "- 本结论针对当前主线使用的 `5s` 窗长、`2.5s` 步长窗口口径。",
    ]

    output_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
