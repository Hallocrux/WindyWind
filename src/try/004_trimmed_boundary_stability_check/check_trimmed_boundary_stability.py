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

from src.current.data_loading import (  # noqa: E402
    DatasetRecord,
    get_common_signal_columns,
    load_clean_signal_frame,
    scan_dataset_records,
)
from src.current.features import WindowConfig, build_case_feature_frame  # noqa: E402

SEGMENT_LABELS = ("start", "middle", "end")
DEFAULT_OUTPUT_DIR = Path("outputs/try/004_trimmed_boundary_stability_check")


@dataclass(frozen=True)
class SegmentBounds:
    label: str
    start_sec: float
    end_sec: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="检查裁剪首尾缺失段后的边界稳定性。")
    parser.add_argument("--segment-seconds", type=float, default=15.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = scan_dataset_records()
    common_columns = get_common_signal_columns(records)
    config = WindowConfig()

    segment_rows: list[dict[str, object]] = []
    distance_rows: list[dict[str, object]] = []
    stationarity_rows: list[dict[str, object]] = []

    for record in records:
        cleaned = load_clean_signal_frame(record, common_columns)
        trimmed = trim_edge_missing_rows(cleaned)
        feature_df = build_case_feature_frame(record, trimmed, config)
        feature_df = add_relative_time_columns(feature_df, trimmed)
        bounds = build_segment_bounds(trimmed, args.segment_seconds)
        feature_df["segment"] = assign_segments(feature_df, bounds)
        segment_df = feature_df[feature_df["segment"].notna()].copy()
        if segment_df.empty:
            continue

        segment_rows.extend(summarize_segments(record, segment_df).to_dict("records"))
        distance_rows.extend(summarize_distances(record, segment_df).to_dict("records"))
        stationarity_rows.extend(summarize_stationarity(record, segment_df).to_dict("records"))

    segment_summary = pd.DataFrame(segment_rows).sort_values(["case_id", "segment"]).reset_index(drop=True)
    distance_summary = pd.DataFrame(distance_rows).sort_values("case_id").reset_index(drop=True)
    stationarity_summary = pd.DataFrame(stationarity_rows).sort_values("case_id").reset_index(drop=True)

    segment_summary.to_csv(args.output_dir / "trimmed_segment_summary.csv", index=False, encoding="utf-8-sig")
    distance_summary.to_csv(args.output_dir / "trimmed_distance_summary.csv", index=False, encoding="utf-8-sig")
    stationarity_summary.to_csv(args.output_dir / "trimmed_stationarity_summary.csv", index=False, encoding="utf-8-sig")

    create_plot(segment_summary, distance_summary, stationarity_summary, args.output_dir / "trimmed_boundary_overview.png")
    write_summary(segment_summary, distance_summary, stationarity_summary, args.output_dir / "summary.md", args.segment_seconds)


def trim_edge_missing_rows(cleaned: pd.DataFrame) -> pd.DataFrame:
    keep_mask = (
        (cleaned["__in_leading_missing_block"] == 0)
        & (cleaned["__in_trailing_missing_block"] == 0)
    )
    trimmed = cleaned.loc[keep_mask].copy().reset_index(drop=True)
    if trimmed.empty:
        raise ValueError("裁剪首尾连续缺失段后无数据可用。")
    return trimmed


def add_relative_time_columns(feature_df: pd.DataFrame, signal_df: pd.DataFrame) -> pd.DataFrame:
    origin = signal_df["time"].iloc[0]
    feature_df = feature_df.copy()
    feature_df["start_sec"] = (feature_df["start_time"] - origin).dt.total_seconds()
    feature_df["end_sec"] = (feature_df["end_time"] - origin).dt.total_seconds()
    feature_df["center_sec"] = (feature_df["start_sec"] + feature_df["end_sec"]) / 2.0
    return feature_df


def build_segment_bounds(signal_df: pd.DataFrame, segment_seconds: float) -> list[SegmentBounds]:
    duration = float((signal_df["time"].iloc[-1] - signal_df["time"].iloc[0]).total_seconds())
    middle_start = max(0.0, duration / 2.0 - segment_seconds / 2.0)
    middle_end = min(duration, middle_start + segment_seconds)
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
        for bound in bounds:
            if row["start_sec"] >= bound.start_sec and row["end_sec"] <= bound.end_sec:
                label = bound.label
                break
        labels.append(label)
    return pd.Series(labels, index=feature_df.index, dtype="object")


def summarize_segments(record: DatasetRecord, segment_df: pd.DataFrame) -> pd.DataFrame:
    rms_cols = [c for c in segment_df.columns if c.endswith("__rms")]
    peak_cols = [c for c in segment_df.columns if c.endswith("__fft_peak_freq")]
    energy_cols = [c for c in segment_df.columns if c.endswith("__fft_total_energy")]
    rows: list[dict[str, object]] = []
    for label in SEGMENT_LABELS:
        block = segment_df[segment_df["segment"] == label]
        rows.append(
            {
                "case_id": record.case_id,
                "file_name": record.file_name,
                "segment": label,
                "window_count": int(len(block)),
                "dirty_window_ratio": float((block["raw_missing_ratio"] > 0).mean()) if not block.empty else np.nan,
                "avg_rms": float(block[rms_cols].mean(axis=1).mean()) if not block.empty else np.nan,
                "avg_peak_freq": float(block[peak_cols].mean(axis=1).mean()) if not block.empty else np.nan,
                "avg_energy": float(block[energy_cols].mean(axis=1).mean()) if not block.empty else np.nan,
                "cv_rms": coefficient_of_variation(block[rms_cols].mean(axis=1)) if not block.empty else np.nan,
                "cv_peak_freq": coefficient_of_variation(block[peak_cols].mean(axis=1)) if not block.empty else np.nan,
            }
        )
    return pd.DataFrame(rows)


def summarize_distances(record: DatasetRecord, segment_df: pd.DataFrame) -> pd.DataFrame:
    vibration_columns = [c for c in segment_df.columns if "__" in c and not c.startswith("__")]
    scaled = StandardScaler().fit_transform(segment_df[vibration_columns])
    scaled_df = pd.DataFrame(scaled, columns=vibration_columns, index=segment_df.index)
    scaled_df["segment"] = segment_df["segment"].to_numpy()

    centroid_map: dict[str, np.ndarray] = {}
    dispersion_map: dict[str, float] = {}
    for label in SEGMENT_LABELS:
        block = scaled_df[scaled_df["segment"] == label].drop(columns=["segment"])
        if block.empty:
            centroid_map[label] = np.full(len(vibration_columns), np.nan)
            dispersion_map[label] = np.nan
            continue
        centroid = block.mean(axis=0).to_numpy(dtype=float)
        centroid_map[label] = centroid
        dispersion_map[label] = float(np.linalg.norm(block.to_numpy(dtype=float) - centroid, axis=1).mean())

    start_mid = safe_distance(centroid_map["start"], centroid_map["middle"])
    end_mid = safe_distance(centroid_map["end"], centroid_map["middle"])
    return pd.DataFrame(
        [
            {
                "case_id": record.case_id,
                "file_name": record.file_name,
                "start_middle_distance": start_mid,
                "end_middle_distance": end_mid,
                "start_middle_vs_within": safe_ratio(start_mid, np.nanmean([dispersion_map["start"], dispersion_map["middle"]])),
                "end_middle_vs_within": safe_ratio(end_mid, np.nanmean([dispersion_map["end"], dispersion_map["middle"]])),
            }
        ]
    )


def summarize_stationarity(record: DatasetRecord, segment_df: pd.DataFrame) -> pd.DataFrame:
    rms_cols = [c for c in segment_df.columns if c.endswith("__rms")]
    peak_cols = [c for c in segment_df.columns if c.endswith("__fft_peak_freq")]
    mean_rms = segment_df[rms_cols].mean(axis=1)
    mean_peak = segment_df[peak_cols].mean(axis=1)

    rows = []
    for label in ("start", "end"):
        block = segment_df[segment_df["segment"] == label].copy()
        if block.empty:
            rows.append(
                {
                    "case_id": record.case_id,
                    "file_name": record.file_name,
                    "segment": label,
                    "rms_slope_per_sec": np.nan,
                    "peak_freq_slope_per_sec": np.nan,
                    "rms_start_to_end_ratio": np.nan,
                    "peak_freq_start_to_end_ratio": np.nan,
                }
            )
            continue

        rms_series = mean_rms.loc[block.index].to_numpy(dtype=float)
        peak_series = mean_peak.loc[block.index].to_numpy(dtype=float)
        times = block["center_sec"].to_numpy(dtype=float)
        rows.append(
            {
                "case_id": record.case_id,
                "file_name": record.file_name,
                "segment": label,
                "rms_slope_per_sec": fit_slope(times, rms_series),
                "peak_freq_slope_per_sec": fit_slope(times, peak_series),
                "rms_start_to_end_ratio": safe_ratio(rms_series[-1], rms_series[0]),
                "peak_freq_start_to_end_ratio": safe_ratio(peak_series[-1], peak_series[0]),
            }
        )
    return pd.DataFrame(rows)


def coefficient_of_variation(series: pd.Series) -> float:
    values = series.to_numpy(dtype=float)
    mean = float(np.mean(values))
    std = float(np.std(values))
    if mean == 0.0:
        return np.nan
    return float(std / mean)


def fit_slope(times: np.ndarray, values: np.ndarray) -> float:
    if len(times) < 2:
        return np.nan
    centered = times - np.mean(times)
    denom = float(np.sum(centered**2))
    if denom == 0.0:
        return np.nan
    return float(np.sum(centered * (values - np.mean(values))) / denom)


def safe_distance(left: np.ndarray, right: np.ndarray) -> float:
    if np.isnan(left).any() or np.isnan(right).any():
        return np.nan
    return float(np.linalg.norm(left - right))


def safe_ratio(value: float, base: float) -> float:
    if np.isnan(value) or np.isnan(base) or base == 0.0:
        return np.nan
    return float(value / base)


def create_plot(
    segment_summary: pd.DataFrame,
    distance_summary: pd.DataFrame,
    stationarity_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    segment_stats = (
        segment_summary.groupby("segment", sort=False)[["dirty_window_ratio", "cv_rms", "cv_peak_freq"]]
        .mean()
        .reindex(SEGMENT_LABELS)
    )
    segment_stats.plot(kind="bar", ax=axes[0])
    axes[0].set_title("Trimmed segment stability")
    axes[0].tick_params(axis="x", rotation=0)

    distance_stats = distance_summary.set_index("case_id")[["start_middle_vs_within", "end_middle_vs_within"]]
    distance_stats.plot(kind="bar", ax=axes[1])
    axes[1].axhline(1.0, color="black", linestyle="--", linewidth=1)
    axes[1].set_title("Trimmed boundary vs middle separation")

    pivot = stationarity_summary.pivot(index="case_id", columns="segment", values="rms_slope_per_sec")
    pivot.plot(kind="bar", ax=axes[2])
    axes[2].axhline(0.0, color="black", linestyle="--", linewidth=1)
    axes[2].set_title("Boundary RMS trend after trimming")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_summary(
    segment_summary: pd.DataFrame,
    distance_summary: pd.DataFrame,
    stationarity_summary: pd.DataFrame,
    output_path: Path,
    segment_seconds: float,
) -> None:
    dirty = (
        segment_summary.groupby("segment", sort=False)["dirty_window_ratio"]
        .mean()
        .reindex(SEGMENT_LABELS)
    )
    start_sep = distance_summary["start_middle_vs_within"]
    end_sep = distance_summary["end_middle_vs_within"]
    start_stationarity = stationarity_summary[stationarity_summary["segment"] == "start"]
    end_stationarity = stationarity_summary[stationarity_summary["segment"] == "end"]

    stable_start = (start_sep <= 1.0).mean() >= 0.5 and start_stationarity["rms_start_to_end_ratio"].between(0.9, 1.1).mean() >= 0.5
    stable_end = (end_sep <= 1.0).mean() >= 0.5 and end_stationarity["rms_start_to_end_ratio"].between(0.9, 1.1).mean() >= 0.5

    lines = [
        "# 裁剪后边界稳定性复核结论",
        "",
        f"- 分段时长：`{segment_seconds:.1f}s`",
        f"- 工况数：`{segment_summary['case_id'].nunique()}`",
        "",
        "## 裁剪后质量",
        "",
        f"- 开始段平均脏窗口占比：`{dirty.loc['start']:.2%}`",
        f"- 中段平均脏窗口占比：`{dirty.loc['middle']:.2%}`",
        f"- 结束段平均脏窗口占比：`{dirty.loc['end']:.2%}`",
        f"- `start vs middle` 相对分离度中位数：`{start_sep.median():.3f}`",
        f"- `end vs middle` 相对分离度中位数：`{end_sep.median():.3f}`",
        f"- 开始段 RMS 首尾比落在 `[0.9, 1.1]` 的工况数：`{int(start_stationarity['rms_start_to_end_ratio'].between(0.9, 1.1).sum())}/{len(start_stationarity)}`",
        f"- 结束段 RMS 首尾比落在 `[0.9, 1.1]` 的工况数：`{int(end_stationarity['rms_start_to_end_ratio'].between(0.9, 1.1).sum())}/{len(end_stationarity)}`",
        "",
        "## 判断",
        "",
        f"- 裁剪后开头：{'更接近稳定段' if stable_start else '仍不能视为普遍稳定'}",
        f"- 裁剪后结尾：{'更接近稳定段' if stable_end else '仍不能视为普遍稳定'}",
        "",
        "## 说明",
        "",
        "- 若边界与中段的相对分离度仍大于 `1.0`，说明它们在特征空间中仍明显偏离稳态中段。",
        "- 若边界段 RMS 首尾比接近 `1.0` 且斜率接近 `0`，更支持其已经进入相对稳定状态。",
        "- 本探索用于回答“删去首尾连续缺失段后，是否还残留明显启动/停机型非稳态”。",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
