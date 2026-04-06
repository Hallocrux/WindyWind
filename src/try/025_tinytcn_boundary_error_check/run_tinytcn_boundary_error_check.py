from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
TRY009_ROOT = REPO_ROOT / "src" / "try" / "009_phase1_feature_groups"
TRY012_ROOT = REPO_ROOT / "src" / "try" / "012_phase3_end_to_end_shortlist"
TRY013_ROOT = REPO_ROOT / "src" / "try" / "013_phase3_cnn_tcn_smoke"
for path in (REPO_ROOT, TRY009_ROOT, TRY012_ROOT, TRY013_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from src.current.data_loading import get_common_signal_columns, load_clean_signal_frame, scan_dataset_records
from src.current.features import WindowConfig

from phase3_cnn_tcn_lib import (
    TorchTrainConfig,
    build_torch_model,
    normalize_windows_by_channel,
    train_torch_model,
)
from phase3_end_to_end_lib import build_raw_window_dataset

TRY_NAME = "025_tinytcn_boundary_error_check"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "try" / TRY_NAME
TARGET_CASE_IDS = [1, 3, 17, 18]
SEGMENT_LABELS = ("start", "middle", "end")


@dataclass(frozen=True)
class SegmentBounds:
    label: str
    start_sec: float
    end_sec: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="检查 TinyTCN 在边界窗口上的误差分布。")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_ROOT,
        help="输出目录，默认写到 outputs/try/025_tinytcn_boundary_error_check。",
    )
    parser.add_argument(
        "--segment-seconds",
        type=float,
        default=15.0,
        help="start/middle/end 每段使用的时长，单位秒。",
    )
    parser.add_argument(
        "--case-ids",
        nargs="+",
        type=int,
        default=TARGET_CASE_IDS,
        help="目标工况列表。",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--permutation-samples",
        type=int,
        default=20000,
        help="当组合数过大时使用的 Monte Carlo 置换次数。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    records = scan_dataset_records()
    common_signal_columns = get_common_signal_columns(records)
    cleaned_signal_frames = {
        record.case_id: load_clean_signal_frame(record, common_signal_columns)
        for record in records
    }

    raw_dataset = build_raw_window_dataset(records, cleaned_signal_frames, WindowConfig())
    train_config = TorchTrainConfig(
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
    )
    target_cases = sorted(set(args.case_ids))
    prediction_frame = evaluate_tinytcn_for_target_cases(
        dataset=raw_dataset,
        target_case_ids=target_cases,
        train_config=train_config,
        random_seed=args.random_seed,
    )
    target_window_df = build_target_window_frame(
        prediction_frame=prediction_frame,
        cleaned_signal_frames=cleaned_signal_frames,
        target_cases=target_cases,
        segment_seconds=args.segment_seconds,
    )
    target_segment_df = target_window_df[target_window_df["segment"].isin(SEGMENT_LABELS)].copy()

    segment_summary_df = summarize_segment_errors(target_segment_df)
    tests_df = build_segment_tests(
        target_segment_df,
        permutation_samples=args.permutation_samples,
        random_seed=args.random_seed,
    )

    target_window_df.to_csv(
        output_dir / "target_case_window_level_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    segment_summary_df.to_csv(
        output_dir / "target_case_segment_error_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    tests_df.to_csv(
        output_dir / "target_case_segment_tests.csv",
        index=False,
        encoding="utf-8-sig",
    )

    create_error_over_time_plot(
        target_window_df,
        output_dir / "target_case_error_over_time.png",
    )
    create_error_boxplot(
        target_segment_df,
        output_dir / "target_case_error_boxplot.png",
    )
    write_summary_markdown(
        output_path=output_dir / "summary.md",
        target_window_df=target_window_df,
        segment_summary_df=segment_summary_df,
        tests_df=tests_df,
        segment_seconds=args.segment_seconds,
    )

    print("TinyTCN 边界窗口误差检查已完成。")
    print(f"输出目录: {output_dir}")
    print(f"目标工况: {target_cases}")


def evaluate_tinytcn_for_target_cases(
    dataset,
    target_case_ids: list[int],
    train_config: TorchTrainConfig,
    random_seed: int,
) -> pd.DataFrame:
    labeled_mask = dataset.meta_df["wind_speed"].notna().to_numpy()
    labeled_meta = dataset.meta_df.loc[labeled_mask].reset_index(drop=True)
    labeled_windows = dataset.windows[labeled_mask]
    y_all = labeled_meta["wind_speed"].to_numpy(dtype=np.float32, copy=False)
    case_values = labeled_meta["case_id"].to_numpy(dtype=int, copy=False)
    device = torch.device("cpu")

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    prediction_frames: list[pd.DataFrame] = []
    for case_id in target_case_ids:
        valid_idx = np.flatnonzero(case_values == case_id)
        if valid_idx.size == 0:
            continue
        train_idx = np.flatnonzero(case_values != case_id)

        X_train = labeled_windows[train_idx]
        X_valid = labeled_windows[valid_idx]
        y_train = y_all[train_idx]
        y_valid = y_all[valid_idx]

        X_train_norm, X_valid_norm = normalize_windows_by_channel(X_train, X_valid)
        model = build_torch_model("TinyTCN", in_channels=X_train.shape[1]).to(device)
        train_torch_model(
            model=model,
            X_train=X_train_norm,
            y_train=y_train,
            X_valid=X_valid_norm,
            y_valid=y_valid,
            config=train_config,
            device=device,
        )
        with torch.no_grad():
            pred = model(torch.from_numpy(X_valid_norm).to(device)).cpu().numpy()

        valid_df = labeled_meta.iloc[valid_idx][
            ["case_id", "file_name", "window_index", "start_time", "end_time", "wind_speed"]
        ].copy()
        valid_df = valid_df.rename(columns={"wind_speed": "true_wind_speed"})
        valid_df["pred_wind_speed"] = pred
        prediction_frames.append(valid_df)

    if not prediction_frames:
        raise ValueError("未生成任何目标工况的窗口级预测。")
    return pd.concat(prediction_frames, ignore_index=True)


def build_target_window_frame(
    prediction_frame: pd.DataFrame,
    cleaned_signal_frames: dict[int, pd.DataFrame],
    target_cases: list[int],
    segment_seconds: float,
) -> pd.DataFrame:
    window_df = prediction_frame[prediction_frame["case_id"].isin(target_cases)].copy()
    window_df["signed_error"] = (
        window_df["pred_wind_speed"] - window_df["true_wind_speed"]
    )
    window_df["abs_error"] = window_df["signed_error"].abs()
    window_df["segment"] = "other"
    window_df["case_duration_sec"] = np.nan
    window_df["start_sec"] = np.nan
    window_df["end_sec"] = np.nan
    window_df["center_sec"] = np.nan

    for case_id in target_cases:
        case_mask = window_df["case_id"] == case_id
        if not case_mask.any():
            continue
        cleaned_signal = cleaned_signal_frames[case_id]
        origin = cleaned_signal["time"].iloc[0]
        duration_sec = float(
            (cleaned_signal["time"].iloc[-1] - origin).total_seconds()
        )
        bounds = build_segment_bounds(cleaned_signal, segment_seconds)

        case_df = window_df.loc[case_mask].copy()
        start_sec = (case_df["start_time"] - origin).dt.total_seconds()
        end_sec = (case_df["end_time"] - origin).dt.total_seconds()
        center_sec = (start_sec + end_sec) / 2.0
        segments = assign_segments(start_sec.to_numpy(), end_sec.to_numpy(), bounds)

        window_df.loc[case_mask, "case_duration_sec"] = duration_sec
        window_df.loc[case_mask, "start_sec"] = start_sec.to_numpy()
        window_df.loc[case_mask, "end_sec"] = end_sec.to_numpy()
        window_df.loc[case_mask, "center_sec"] = center_sec.to_numpy()
        window_df.loc[case_mask, "segment"] = segments

    return window_df.sort_values(["case_id", "window_index"]).reset_index(drop=True)


def build_segment_bounds(
    cleaned_signal: pd.DataFrame,
    segment_seconds: float,
) -> list[SegmentBounds]:
    duration = float(
        (cleaned_signal["time"].iloc[-1] - cleaned_signal["time"].iloc[0]).total_seconds()
    )
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


def assign_segments(
    start_sec: np.ndarray,
    end_sec: np.ndarray,
    bounds: list[SegmentBounds],
) -> list[str]:
    labels: list[str] = []
    for left, right in zip(start_sec, end_sec, strict=True):
        label = "other"
        for segment in bounds:
            if left >= segment.start_sec and right <= segment.end_sec:
                label = segment.label
                break
        labels.append(label)
    return labels


def summarize_segment_errors(target_segment_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for case_id, case_df in target_segment_df.groupby("case_id", sort=True):
        file_name = str(case_df["file_name"].iloc[0])
        true_wind_speed = float(case_df["true_wind_speed"].iloc[0])
        for segment in SEGMENT_LABELS:
            block = case_df[case_df["segment"] == segment]
            rows.append(
                {
                    "case_id": case_id,
                    "file_name": file_name,
                    "true_wind_speed": true_wind_speed,
                    "segment": segment,
                    "window_count": int(len(block)),
                    "mean_abs_error": float(block["abs_error"].mean()),
                    "median_abs_error": float(block["abs_error"].median()),
                    "std_abs_error": float(block["abs_error"].std(ddof=0)),
                    "mean_signed_error": float(block["signed_error"].mean()),
                    "mean_pred_wind_speed": float(block["pred_wind_speed"].mean()),
                    "median_center_sec": float(block["center_sec"].median()),
                }
            )
    return pd.DataFrame(rows).sort_values(["case_id", "segment"]).reset_index(drop=True)


def build_segment_tests(
    target_segment_df: pd.DataFrame,
    permutation_samples: int,
    random_seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for case_id, case_df in target_segment_df.groupby("case_id", sort=True):
        for left_label, right_label, pair_name in (
            ("start", "middle", "start_vs_middle"),
            ("end", "middle", "end_vs_middle"),
            ("boundary", "middle", "boundary_vs_middle"),
        ):
            if left_label == "boundary":
                left_values = case_df[
                    case_df["segment"].isin(["start", "end"])
                ]["abs_error"].to_numpy(dtype=float)
            else:
                left_values = case_df[
                    case_df["segment"] == left_label
                ]["abs_error"].to_numpy(dtype=float)
            right_values = case_df[
                case_df["segment"] == right_label
            ]["abs_error"].to_numpy(dtype=float)

            test_result = permutation_test_mean_diff(
                left_values,
                right_values,
                alternative="greater",
                random_seed=random_seed + case_id,
                max_monte_carlo_samples=permutation_samples,
            )
            rows.append(
                {
                    "case_id": case_id,
                    "comparison": pair_name,
                    "left_group": left_label,
                    "right_group": right_label,
                    "left_count": int(len(left_values)),
                    "right_count": int(len(right_values)),
                    "left_mean_abs_error": float(np.mean(left_values)),
                    "right_mean_abs_error": float(np.mean(right_values)),
                    "observed_mean_diff": float(np.mean(left_values) - np.mean(right_values)),
                    "p_value_one_sided": test_result["p_value"],
                    "method": test_result["method"],
                    "permutation_count": int(test_result["permutation_count"]),
                    "is_significant_0p05": int(test_result["p_value"] < 0.05),
                }
            )
    return pd.DataFrame(rows).sort_values(["case_id", "comparison"]).reset_index(drop=True)


def permutation_test_mean_diff(
    left_values: np.ndarray,
    right_values: np.ndarray,
    alternative: str,
    random_seed: int,
    max_monte_carlo_samples: int,
) -> dict[str, object]:
    if left_values.size == 0 or right_values.size == 0:
        return {
            "p_value": np.nan,
            "method": "invalid",
            "permutation_count": 0,
        }

    observed = float(np.mean(left_values) - np.mean(right_values))
    pooled = np.concatenate([left_values, right_values])
    left_size = int(left_values.size)
    total_size = int(pooled.size)
    combination_count = math.comb(total_size, left_size)

    if combination_count <= 50000:
        diffs = []
        all_indices = np.arange(total_size)
        for left_index_tuple in combinations(range(total_size), left_size):
            left_idx = np.array(left_index_tuple, dtype=int)
            mask = np.ones(total_size, dtype=bool)
            mask[left_idx] = False
            right_idx = all_indices[mask]
            diffs.append(float(np.mean(pooled[left_idx]) - np.mean(pooled[right_idx])))
        permuted = np.array(diffs, dtype=float)
        method = "exact"
    else:
        rng = np.random.default_rng(random_seed)
        permuted = np.empty(max_monte_carlo_samples, dtype=float)
        for i in range(max_monte_carlo_samples):
            shuffled = rng.permutation(pooled)
            left = shuffled[:left_size]
            right = shuffled[left_size:]
            permuted[i] = float(np.mean(left) - np.mean(right))
        method = "monte_carlo"

    if alternative == "greater":
        p_value = float(np.mean(permuted >= observed))
    elif alternative == "less":
        p_value = float(np.mean(permuted <= observed))
    else:
        raise ValueError(f"未知 alternative: {alternative}")

    return {
        "p_value": p_value,
        "method": method,
        "permutation_count": int(len(permuted)),
    }


def create_error_over_time_plot(target_window_df: pd.DataFrame, output_path: Path) -> None:
    case_ids = sorted(target_window_df["case_id"].unique())
    fig, axes = plt.subplots(len(case_ids), 1, figsize=(12, 3.2 * len(case_ids)), sharex=False)
    if len(case_ids) == 1:
        axes = [axes]

    color_map = {
        "start": "#d73027",
        "middle": "#1a9850",
        "end": "#4575b4",
        "other": "#aaaaaa",
    }
    for ax, case_id in zip(axes, case_ids, strict=True):
        case_df = target_window_df[target_window_df["case_id"] == case_id].copy()
        file_name = str(case_df["file_name"].iloc[0])
        ax.plot(case_df["center_sec"], case_df["abs_error"], color="#444444", linewidth=1.0)
        for segment, block in case_df.groupby("segment", sort=False):
            ax.scatter(
                block["center_sec"],
                block["abs_error"],
                label=segment,
                s=28,
                color=color_map.get(segment, "#777777"),
                alpha=0.9,
            )
        ax.set_title(f"case {case_id} | {file_name}")
        ax.set_xlabel("window center sec")
        ax.set_ylabel("abs error")
        ax.legend(loc="upper right", ncols=4, fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def create_error_boxplot(target_segment_df: pd.DataFrame, output_path: Path) -> None:
    case_ids = sorted(target_segment_df["case_id"].unique())
    fig, axes = plt.subplots(1, len(case_ids), figsize=(4.2 * len(case_ids), 4.5), sharey=True)
    if len(case_ids) == 1:
        axes = [axes]

    for ax, case_id in zip(axes, case_ids, strict=True):
        case_df = target_segment_df[target_segment_df["case_id"] == case_id]
        data = [
            case_df[case_df["segment"] == segment]["abs_error"].to_numpy(dtype=float)
            for segment in SEGMENT_LABELS
        ]
        ax.boxplot(data, labels=SEGMENT_LABELS, showmeans=True)
        ax.set_title(f"case {case_id}")
        ax.set_xlabel("segment")
        ax.set_ylabel("abs error")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_summary_markdown(
    output_path: Path,
    target_window_df: pd.DataFrame,
    segment_summary_df: pd.DataFrame,
    tests_df: pd.DataFrame,
    segment_seconds: float,
) -> None:
    overall_rows = []
    for case_id, case_df in segment_summary_df.groupby("case_id", sort=True):
        pivot = case_df.set_index("segment")
        start_mean = float(pivot.loc["start", "mean_abs_error"])
        middle_mean = float(pivot.loc["middle", "mean_abs_error"])
        end_mean = float(pivot.loc["end", "mean_abs_error"])
        overall_rows.append(
            {
                "case_id": case_id,
                "start_mean": start_mean,
                "middle_mean": middle_mean,
                "end_mean": end_mean,
                "boundary_mean": float(np.mean([start_mean, end_mean])),
            }
        )
    overall_df = pd.DataFrame(overall_rows).sort_values("case_id")
    boundary_worse_count = int(
        (overall_df["boundary_mean"] > overall_df["middle_mean"]).sum()
    )
    significant_count = int(
        tests_df[
            (tests_df["comparison"] == "boundary_vs_middle")
            & (tests_df["is_significant_0p05"] == 1)
        ].shape[0]
    )

    lines = [
        "# TinyTCN 边界窗口误差检查结论",
        "",
        f"- 目标工况：`{sorted(target_window_df['case_id'].unique().tolist())}`",
        f"- 窗口口径：`50Hz / 5s / 2.5s`",
        f"- 分段口径：`start / middle / end = {segment_seconds:.1f}s`",
        f"- 检查窗口总数：`{len(target_window_df)}`",
        f"- 参与分段统计的窗口数：`{int(target_window_df['segment'].isin(SEGMENT_LABELS).sum())}`",
        "",
        "## 每工况边界是否更差",
        "",
    ]
    for _, row in overall_df.iterrows():
        lines.append(
            f"- `工况{int(row['case_id'])}`: start=`{row['start_mean']:.4f}`, middle=`{row['middle_mean']:.4f}`, end=`{row['end_mean']:.4f}`, boundary_mean=`{row['boundary_mean']:.4f}`"
        )

    lines.extend(
        [
            "",
            "## 显著性检查",
            "",
            f"- `boundary_vs_middle` 中，边界均值误差高于中段的工况数：`{boundary_worse_count}/{len(overall_df)}`",
            f"- `boundary_vs_middle` 在 `p < 0.05` 下显著的工况数：`{significant_count}/{len(overall_df)}`",
            "",
        ]
    )
    for case_id in sorted(target_window_df["case_id"].unique()):
        case_tests = tests_df[tests_df["case_id"] == case_id].copy()
        lines.append(f"### 工况{case_id}")
        lines.append("")
        for _, row in case_tests.iterrows():
            lines.append(
                f"- `{row['comparison']}`: left_mean=`{row['left_mean_abs_error']:.4f}`, right_mean=`{row['right_mean_abs_error']:.4f}`, diff=`{row['observed_mean_diff']:.4f}`, p=`{row['p_value_one_sided']:.4f}`, method=`{row['method']}`"
            )
        lines.append("")

    lines.extend(
        [
            "## 说明",
            "",
            "- 显著性检验比较的是平均绝对误差差值，备择假设是“边界段误差大于中段误差”。",
            "- 这里的证据只回答 `TinyTCN + 当前 5s 窗口口径` 下的边界误差问题，不自动外推到其他模型或其他窗长。",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
