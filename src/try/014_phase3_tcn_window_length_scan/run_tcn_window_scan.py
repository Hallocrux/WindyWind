from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
TRY009_ROOT = REPO_ROOT / "src" / "try" / "009_phase1_feature_groups"
TRY012_ROOT = REPO_ROOT / "src" / "try" / "012_phase3_end_to_end_shortlist"
TRY013_ROOT = REPO_ROOT / "src" / "try" / "013_phase3_cnn_tcn_smoke"
TRY014_ROOT = REPO_ROOT / "src" / "try" / "014_phase3_tcn_window_length_scan"
for path in (REPO_ROOT, TRY009_ROOT, TRY012_ROOT, TRY013_ROOT, TRY014_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from src.current.data_loading import get_common_signal_columns, load_clean_signal_frame, scan_dataset_records
from src.current.features import WindowConfig

from phase3_cnn_tcn_lib import TorchTrainConfig, evaluate_torch_model_loco, summarize_predictions
from phase3_end_to_end_lib import build_raw_window_dataset

TRY_NAME = "014_phase3_tcn_window_length_scan"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "try" / TRY_NAME
DEV_CASE_IDS = [1, 2, 3, 5, 15, 16]
WINDOW_CONFIGS = [
    ("2s", WindowConfig(sampling_rate=50.0, window_size=100, step_size=50)),
    ("4s", WindowConfig(sampling_rate=50.0, window_size=200, step_size=100)),
    ("5s", WindowConfig(sampling_rate=50.0, window_size=250, step_size=125)),
    ("8s", WindowConfig(sampling_rate=50.0, window_size=400, step_size=200)),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 TinyTCN 窗长扫描。")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_ROOT,
        help="输出目录，默认写到 outputs/try/014_phase3_tcn_window_length_scan。",
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
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
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
        "# TinyTCN 窗长扫描结论",
        "",
        f"- 运行工况：`{[record.case_id for record in records]}`",
        f"- 最优窗长：`{best_row['window_label']}`",
        f"- 最优 case_mae：`{float(best_row['case_mae']):.4f}`",
        f"- 最优 case_rmse：`{float(best_row['case_rmse']):.4f}`",
        "",
        "## 全部窗长",
        "",
    ]
    for _, row in summary_df.iterrows():
        lines.append(
            f"- `TinyTCN @ {row['window_label']}`: case_mae=`{float(row['case_mae']):.4f}`, case_rmse=`{float(row['case_rmse']):.4f}`, windows=`{int(row['window_count'])}`"
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    train_config = TorchTrainConfig(
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
    )

    records = select_records(args)
    common_signal_columns = get_common_signal_columns(records)
    cleaned_signal_frames = {
        record.case_id: load_clean_signal_frame(record, common_signal_columns)
        for record in records
    }

    summary_rows: list[dict[str, object]] = []
    case_frames: list[pd.DataFrame] = []
    for window_label, window_config in WINDOW_CONFIGS:
        raw_dataset = build_raw_window_dataset(records, cleaned_signal_frames, window_config)
        prediction_frame = evaluate_torch_model_loco(raw_dataset, "TinyTCN", train_config)
        summary_row, case_df = summarize_predictions(prediction_frame, f"TinyTCN@{window_label}")
        summary_row["window_label"] = window_label
        summary_row["window_size"] = window_config.window_size
        summary_row["step_size"] = window_config.step_size
        summary_row["window_count"] = int(len(raw_dataset.meta_df))
        case_df["window_label"] = window_label
        case_frames.append(case_df)
        summary_rows.append(summary_row)

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["case_mae", "case_rmse", "window_label"]
    ).reset_index(drop=True)
    case_df = pd.concat(case_frames, ignore_index=True)

    summary_df.to_csv(output_dir / "tcn_window_scan_summary.csv", index=False, encoding="utf-8-sig")
    case_df.to_csv(output_dir / "tcn_window_scan_case_level_predictions.csv", index=False, encoding="utf-8-sig")
    write_summary_markdown(output_dir, summary_df, records)

    print("TinyTCN 窗长扫描已完成。")
    print(f"输出目录: {output_dir}")
    print(f"运行工况: {[record.case_id for record in records]}")
    print(f"最优窗长: {summary_df.iloc[0]['window_label']}")


if __name__ == "__main__":
    main()
