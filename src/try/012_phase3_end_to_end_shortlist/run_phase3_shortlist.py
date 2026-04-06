from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
TRY009_ROOT = REPO_ROOT / "src" / "try" / "009_phase1_feature_groups"
TRY012_ROOT = REPO_ROOT / "src" / "try" / "012_phase3_end_to_end_shortlist"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(TRY009_ROOT) not in sys.path:
    sys.path.insert(0, str(TRY009_ROOT))
if str(TRY012_ROOT) not in sys.path:
    sys.path.insert(0, str(TRY012_ROOT))

from src.current.data_loading import get_common_signal_columns, load_clean_signal_frame, scan_dataset_records
from src.current.features import WindowConfig

from phase3_end_to_end_lib import (
    build_raw_window_dataset,
    evaluate_raw_model_loco,
    evaluate_tabular_reference_loco,
    summarize_predictions,
)

TRY_NAME = "012_phase3_end_to_end_shortlist"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "try" / TRY_NAME
DEV_CASE_IDS = [1, 2, 3, 5, 15, 16]
MODEL_NAMES = [
    "TabularReference_G6_Ridge",
    "RawFlattenRidge",
    "MiniRocketLikeRidge",
    "RawFlattenMLP",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行第三阶段端到端 shortlist。")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_ROOT,
        help="输出目录，默认写到 outputs/try/012_phase3_end_to_end_shortlist。",
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


def write_summary_markdown(output_dir: Path, summary_df: pd.DataFrame, records: list) -> None:
    best_row = summary_df.iloc[0]
    lines = [
        "# 第三阶段端到端 shortlist 结论",
        "",
        f"- 运行工况：`{[record.case_id for record in records]}`",
        f"- 最优模型：`{best_row['model_name']}`",
        f"- 最优 case_mae：`{float(best_row['case_mae']):.4f}`",
        f"- 最优 case_rmse：`{float(best_row['case_rmse']):.4f}`",
        "",
        "## 全部模型",
        "",
    ]
    for _, row in summary_df.iterrows():
        lines.append(
            f"- `{row['model_name']}`: case_mae=`{float(row['case_mae']):.4f}`, case_rmse=`{float(row['case_rmse']):.4f}`"
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
    raw_dataset = build_raw_window_dataset(records, cleaned_signal_frames, WindowConfig())

    summary_rows: list[dict[str, object]] = []
    case_frames: list[pd.DataFrame] = []
    for model_name in MODEL_NAMES:
        if model_name == "TabularReference_G6_Ridge":
            prediction_frame = evaluate_tabular_reference_loco(records, cleaned_signal_frames, WindowConfig())
        else:
            prediction_frame = evaluate_raw_model_loco(raw_dataset, model_name)
        summary_row, case_df = summarize_predictions(prediction_frame, model_name)
        summary_rows.append(summary_row)
        case_frames.append(case_df)

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["case_mae", "case_rmse", "model_name"]
    ).reset_index(drop=True)
    case_df = pd.concat(case_frames, ignore_index=True)

    summary_df.to_csv(output_dir / "phase3_model_summary.csv", index=False, encoding="utf-8-sig")
    case_df.to_csv(output_dir / "phase3_case_level_predictions.csv", index=False, encoding="utf-8-sig")
    write_summary_markdown(output_dir, summary_df, records)

    print("phase 3 shortlist 已完成。")
    print(f"输出目录: {output_dir}")
    print(f"运行工况: {[record.case_id for record in records]}")
    print(f"最优模型: {summary_df.iloc[0]['model_name']}")


if __name__ == "__main__":
    main()
