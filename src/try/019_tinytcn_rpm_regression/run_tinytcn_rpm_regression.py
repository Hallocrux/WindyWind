from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
TRY009_ROOT = REPO_ROOT / "src" / "try" / "009_phase1_feature_groups"
TRY012_ROOT = REPO_ROOT / "src" / "try" / "012_phase3_end_to_end_shortlist"
TRY019_ROOT = REPO_ROOT / "src" / "try" / "019_tinytcn_rpm_regression"
for path in (REPO_ROOT, TRY009_ROOT, TRY012_ROOT, TRY019_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from src.current.data_loading import (
    get_common_signal_columns,
    load_clean_signal_frame,
    scan_dataset_records,
)
from src.current.features import WindowConfig

from phase3_end_to_end_lib import build_raw_window_dataset
from tinytcn_rpm_lib import (
    TorchTrainConfig,
    evaluate_tinytcn_rpm_loco,
    predict_tinytcn_rpm_unlabeled,
    summarize_rpm_predictions,
)

TRY_NAME = "019_tinytcn_rpm_regression"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "try" / TRY_NAME
DEV_CASE_IDS = [1, 2, 3, 5, 15, 16]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 TinyTCN 转速回归。")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_ROOT,
        help="输出目录，默认写到 outputs/try/019_tinytcn_rpm_regression。",
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


def write_summary_markdown(
    output_dir: Path,
    summary_row: dict[str, object],
    records: list,
    unlabeled_df: pd.DataFrame,
) -> None:
    labeled_case_ids = [record.case_id for record in records if record.rpm is not None]
    unlabeled_case_ids = [record.case_id for record in records if record.rpm is None]
    lines = [
        "# TinyTCN 转速回归结论",
        "",
        f"- 运行工况：`{[record.case_id for record in records]}`",
        f"- 带 rpm 标签工况：`{labeled_case_ids}`",
        f"- 无 rpm 标签工况：`{unlabeled_case_ids}`",
        f"- 模型：`{summary_row['model_name']}`",
        f"- case_mae：`{float(summary_row['case_mae']):.4f}`",
        f"- case_rmse：`{float(summary_row['case_rmse']):.4f}`",
        f"- case_mape：`{float(summary_row['case_mape']):.4f}%`",
    ]
    if not unlabeled_df.empty:
        first_row = unlabeled_df.iloc[0]
        lines.extend(
            [
                "",
                "## 无标签工况推理",
                "",
                f"- `{first_row['file_name']}` 预测转速：`{float(first_row['predicted_rpm']):.4f} rpm`",
            ]
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
    raw_dataset = build_raw_window_dataset(records, cleaned_signal_frames, WindowConfig())

    prediction_frame = evaluate_tinytcn_rpm_loco(raw_dataset, train_config)
    summary_row, case_df = summarize_rpm_predictions(prediction_frame, "TinyTCN")
    summary_df = pd.DataFrame([summary_row])
    unlabeled_df = predict_tinytcn_rpm_unlabeled(raw_dataset, train_config)

    summary_df.to_csv(output_dir / "model_summary.csv", index=False, encoding="utf-8-sig")
    case_df.to_csv(output_dir / "case_level_predictions.csv", index=False, encoding="utf-8-sig")
    unlabeled_df.to_csv(output_dir / "unlabeled_predictions.csv", index=False, encoding="utf-8-sig")
    write_summary_markdown(output_dir, summary_row, records, unlabeled_df)

    print("TinyTCN 转速回归已完成。")
    print(f"输出目录: {output_dir}")
    print(f"运行工况: {[record.case_id for record in records]}")
    print(f"case_mae: {float(summary_row['case_mae']):.4f}")
    if not unlabeled_df.empty:
        first_row = unlabeled_df.iloc[0]
        print(f"{first_row['file_name']} 预测转速: {float(first_row['predicted_rpm']):.4f} rpm")


if __name__ == "__main__":
    main()

