from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
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

TRY_NAME = "011_phase1_shortlist_models"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "try" / TRY_NAME
DEV_CASE_IDS = [1, 2, 3, 5, 15, 16]
SHORTLIST_EXPERIMENTS = [
    ("G3_CROSS_CHANNEL", "rpm_free"),
    ("G1_ROBUST_TIME", "rpm_free"),
    ("G6_TIME_FREQ_CROSS", "rpm_free"),
    ("G3_CROSS_CHANNEL", "rpm_aware"),
]
MODEL_NAMES = ["RandomForestRegressor", "HistGradientBoostingRegressor"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行第一阶段 shortlist 树模型比较。")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_ROOT,
        help="输出目录，默认写到 outputs/try/011_phase1_shortlist_models。",
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
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="外层并行 worker 数。",
    )
    parser.add_argument(
        "--rf-n-jobs",
        type=int,
        default=1,
        help="RandomForestRegressor 的 n_jobs 设置。",
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


def build_matrix_cache(feature_df: pd.DataFrame) -> dict[tuple[str, str], tuple[list[str], np.ndarray]]:
    labeled_df = feature_df[feature_df["wind_speed"].notna()].copy()
    cache: dict[tuple[str, str], tuple[list[str], np.ndarray]] = {}
    for group_name, task_mode in SHORTLIST_EXPERIMENTS:
        feature_columns = get_group_feature_columns(feature_df, group_name)
        if task_mode == "rpm_aware":
            feature_columns = [*feature_columns, "rpm"]
        cache[(group_name, task_mode)] = (
            feature_columns,
            labeled_df[feature_columns].to_numpy(dtype=float, copy=False),
        )
    return cache


def run_single_experiment(
    labeled_df: pd.DataFrame,
    split_map: dict[int, tuple[np.ndarray, np.ndarray]],
    matrix: np.ndarray,
    feature_count: int,
    group_name: str,
    task_mode: str,
    model_name: str,
    runtime_config: Phase1RuntimeConfig,
) -> tuple[dict[str, object], pd.DataFrame]:
    prediction_frame = evaluate_loco(
        labeled_df=labeled_df,
        split_map=split_map,
        matrix=matrix,
        model_name=model_name,
        runtime_config=runtime_config,
    )
    experiment_name = f"{model_name}__{group_name}__{task_mode}"
    case_frame = summarize_case_predictions(
        prediction_frame,
        model_name=model_name,
        feature_set=experiment_name,
    )
    case_frame["group_name"] = group_name
    case_frame["task_mode"] = task_mode
    case_frame["feature_count"] = feature_count
    errors = case_frame["pred_mean"] - case_frame["true_wind_speed"]
    summary_row = {
        "experiment_name": experiment_name,
        "group_name": group_name,
        "task_mode": task_mode,
        "model_name": model_name,
        "feature_count": feature_count,
        "case_mae": float(np.mean(np.abs(errors))),
        "case_rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "case_mape": float(
            np.mean(np.abs(errors) / case_frame["true_wind_speed"].to_numpy(dtype=float)) * 100
        ),
    }
    return summary_row, case_frame


def run_shortlist_tree_models(
    feature_df: pd.DataFrame,
    runtime_config: Phase1RuntimeConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    labeled_df = feature_df[feature_df["wind_speed"].notna()].copy()
    split_map = build_loco_split_map(labeled_df)
    matrix_cache = build_matrix_cache(feature_df)
    task_payloads = [
        (group_name, task_mode, model_name)
        for group_name, task_mode in SHORTLIST_EXPERIMENTS
        for model_name in MODEL_NAMES
    ]

    summary_rows: list[dict[str, object]] = []
    case_frames: list[pd.DataFrame] = []
    if runtime_config.max_workers > 1:
        with ThreadPoolExecutor(max_workers=runtime_config.max_workers) as executor:
            futures = [
                executor.submit(
                    run_single_experiment,
                    labeled_df,
                    split_map,
                    matrix_cache[(group_name, task_mode)][1],
                    len(matrix_cache[(group_name, task_mode)][0]),
                    group_name,
                    task_mode,
                    model_name,
                    runtime_config,
                )
                for group_name, task_mode, model_name in task_payloads
            ]
            for future in futures:
                summary_row, case_frame = future.result()
                summary_rows.append(summary_row)
                case_frames.append(case_frame)
    else:
        for group_name, task_mode, model_name in task_payloads:
            summary_row, case_frame = run_single_experiment(
                labeled_df=labeled_df,
                split_map=split_map,
                matrix=matrix_cache[(group_name, task_mode)][1],
                feature_count=len(matrix_cache[(group_name, task_mode)][0]),
                group_name=group_name,
                task_mode=task_mode,
                model_name=model_name,
                runtime_config=runtime_config,
            )
            summary_rows.append(summary_row)
            case_frames.append(case_frame)

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["case_mae", "case_rmse", "model_name", "experiment_name"]
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
        "# 第一阶段 shortlist 树模型结论",
        "",
        f"- 运行工况：`{[record.case_id for record in records]}`",
        f"- 最优模型：`{best_row['model_name']}`",
        f"- 最优组合：`{best_row['experiment_name']}`",
        f"- 最优 case_mae：`{float(best_row['case_mae']):.4f}`",
        f"- 最优 case_rmse：`{float(best_row['case_rmse']):.4f}`",
        "",
        "## 全部实验",
        "",
    ]
    for _, row in summary_df.iterrows():
        lines.append(
            f"- `{row['experiment_name']}`: case_mae=`{float(row['case_mae']):.4f}`, case_rmse=`{float(row['case_rmse']):.4f}`"
        )

    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime_config = Phase1RuntimeConfig(
        max_workers=max(1, args.max_workers),
        rf_n_jobs=args.rf_n_jobs,
    )

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

    summary_df, case_df = run_shortlist_tree_models(feature_df, runtime_config)
    summary_df.to_csv(
        output_dir / "shortlist_tree_models_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    case_df.to_csv(
        output_dir / "shortlist_tree_models_case_level_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    write_summary_markdown(output_dir, summary_df, records)

    print("shortlist 树模型比较已完成。")
    print(f"输出目录: {output_dir}")
    print(f"运行工况: {[record.case_id for record in records]}")
    print(f"最优模型: {summary_df.iloc[0]['model_name']}")


if __name__ == "__main__":
    main()
