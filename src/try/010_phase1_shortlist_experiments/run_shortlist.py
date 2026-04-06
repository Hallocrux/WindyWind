from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
TRY009_ROOT = REPO_ROOT / "src" / "try" / "009_phase1_feature_groups"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(TRY009_ROOT) not in sys.path:
    sys.path.insert(0, str(TRY009_ROOT))

from src.current.data_loading import get_common_signal_columns, load_clean_signal_frame, scan_dataset_records
from src.current.features import WindowConfig

from phase1_feature_groups_lib import (
    Phase1RuntimeConfig,
    build_estimator,
    build_feature_frame,
    build_loco_split_map,
    evaluate_loco,
    get_group_feature_columns,
    summarize_case_predictions,
)

TRY_NAME = "010_phase1_shortlist_experiments"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "try" / TRY_NAME
DEV_CASE_IDS = [1, 2, 3, 5, 15, 16]
SHORTLIST_EXPERIMENTS = [
    ("G3_CROSS_CHANNEL", "rpm_free"),
    ("G1_ROBUST_TIME", "rpm_free"),
    ("G6_TIME_FREQ_CROSS", "rpm_free"),
    ("G3_CROSS_CHANNEL", "rpm_aware"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行第一阶段 shortlist 实验。")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_ROOT,
        help="输出目录，默认写到 outputs/try/010_phase1_shortlist_experiments。",
    )
    parser.add_argument(
        "--mode",
        choices=["dev", "full"],
        default="dev",
        help="运行模式：dev 使用固定小数据集，full 使用全部工况。",
    )
    parser.add_argument(
        "--case-ids",
        nargs="+",
        type=int,
        default=None,
        help="显式指定参与实验的 case_id 列表；指定后优先级高于 --mode。",
    )
    return parser.parse_args()


def select_records(args: argparse.Namespace):
    records = scan_dataset_records()
    if args.case_ids:
        selected = set(args.case_ids)
        return [record for record in records if record.case_id in selected]
    if args.mode == "dev":
        return [record for record in records if record.case_id in set(DEV_CASE_IDS)]
    return records


def build_shortlist_feature_manifest(feature_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group_name, task_mode in SHORTLIST_EXPERIMENTS:
        feature_columns = get_group_feature_columns(feature_df, group_name)
        if task_mode == "rpm_aware":
            feature_columns = [*feature_columns, "rpm"]
        experiment_name = f"Ridge__{group_name}__{task_mode}"
        for order, column in enumerate(feature_columns):
            rows.append(
                {
                    "experiment_name": experiment_name,
                    "group_name": group_name,
                    "task_mode": task_mode,
                    "feature_order": order,
                    "feature_column": column,
                }
            )
    return pd.DataFrame(rows)


def run_shortlist_experiments(feature_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    labeled_df = feature_df[feature_df["wind_speed"].notna()].copy()
    split_map = build_loco_split_map(labeled_df)
    runtime_config = Phase1RuntimeConfig(max_workers=1, rf_n_jobs=1)
    summary_rows: list[dict[str, object]] = []
    case_frames: list[pd.DataFrame] = []

    for group_name, task_mode in SHORTLIST_EXPERIMENTS:
        feature_columns = get_group_feature_columns(feature_df, group_name)
        if task_mode == "rpm_aware":
            feature_columns = [*feature_columns, "rpm"]
        matrix = labeled_df[feature_columns].to_numpy(dtype=float, copy=False)
        prediction_frame = evaluate_loco(
            labeled_df=labeled_df,
            split_map=split_map,
            matrix=matrix,
            model_name="Ridge",
            runtime_config=runtime_config,
        )
        experiment_name = f"Ridge__{group_name}__{task_mode}"
        case_frame = summarize_case_predictions(
            prediction_frame,
            model_name="Ridge",
            feature_set=experiment_name,
        )
        case_frame["group_name"] = group_name
        case_frame["task_mode"] = task_mode
        case_frame["feature_count"] = len(feature_columns)
        errors = case_frame["pred_mean"] - case_frame["true_wind_speed"]
        summary_rows.append(
            {
                "experiment_name": experiment_name,
                "group_name": group_name,
                "task_mode": task_mode,
                "model_name": "Ridge",
                "feature_count": len(feature_columns),
                "case_mae": float(np.mean(np.abs(errors))),
                "case_rmse": float(np.sqrt(np.mean(np.square(errors)))),
                "case_mape": float(
                    np.mean(np.abs(errors) / case_frame["true_wind_speed"].to_numpy(dtype=float)) * 100
                ),
            }
        )
        case_frames.append(case_frame)

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["case_mae", "case_rmse", "feature_count", "experiment_name"]
    ).reset_index(drop=True)
    case_df = pd.concat(case_frames, ignore_index=True)
    return summary_df, case_df


def write_summary_markdown(
    output_dir: Path,
    summary_df: pd.DataFrame,
    records: list,
) -> None:
    best_row = summary_df.iloc[0]
    lines = [
        "# 第一阶段 shortlist 实验结论",
        "",
        f"- 运行工况：`{[record.case_id for record in records]}`",
        f"- 最优组合：`{best_row['experiment_name']}`",
        f"- 最优 case_mae：`{float(best_row['case_mae']):.4f}`",
        f"- 最优 case_rmse：`{float(best_row['case_rmse']):.4f}`",
        "",
        "## 全部实验",
        "",
    ]
    for _, row in summary_df.iterrows():
        lines.append(
            f"- `{row['experiment_name']}`: case_mae=`{float(row['case_mae']):.4f}`, case_rmse=`{float(row['case_rmse']):.4f}`, feature_count=`{int(row['feature_count'])}`"
        )

    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    records = select_records(args)
    common_signal_columns = get_common_signal_columns(records)
    cleaned_signal_frames = {
        record.case_id: load_clean_signal_frame(record, common_signal_columns)
        for record in records
    }
    feature_df = build_feature_frame(
        records=records,
        cleaned_signal_frames=cleaned_signal_frames,
        config=WindowConfig(),
    )

    summary_df, case_df = run_shortlist_experiments(feature_df)
    manifest_df = build_shortlist_feature_manifest(feature_df)

    summary_df.to_csv(
        output_dir / "shortlist_model_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    case_df.to_csv(
        output_dir / "shortlist_case_level_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    manifest_df.to_csv(
        output_dir / "shortlist_feature_manifest.csv",
        index=False,
        encoding="utf-8-sig",
    )
    write_summary_markdown(output_dir, summary_df, records)

    print("shortlist 实验已完成。")
    print(f"输出目录: {output_dir}")
    print(f"运行工况: {[record.case_id for record in records]}")
    print(f"最优组合: {summary_df.iloc[0]['experiment_name']}")


if __name__ == "__main__":
    main()
