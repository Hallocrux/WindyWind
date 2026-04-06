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

from phase3_end_to_end_lib import build_raw_window_dataset
from tinytcn_rpm_lib import (
    TorchTrainConfig,
    evaluate_tinytcn_rpm_loco,
    predict_tinytcn_rpm_unlabeled,
    summarize_rpm_predictions,
)

from src.current.data_loading import (
    get_common_signal_columns,
    load_clean_signal_frame,
    scan_dataset_records,
)
from src.current.features import WindowConfig

TRY_NAME = "024_tinytcn_rpm_fine_window_scan"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "try" / TRY_NAME
WINDOW_CONFIGS = [
    ("2.0s", WindowConfig(sampling_rate=50.0, window_size=100, step_size=50)),
    ("2.5s", WindowConfig(sampling_rate=50.0, window_size=125, step_size=62)),
    ("3.0s", WindowConfig(sampling_rate=50.0, window_size=150, step_size=75)),
    ("3.5s", WindowConfig(sampling_rate=50.0, window_size=175, step_size=88)),
    ("4.0s", WindowConfig(sampling_rate=50.0, window_size=200, step_size=100)),
    ("4.5s", WindowConfig(sampling_rate=50.0, window_size=225, step_size=112)),
    ("5.0s", WindowConfig(sampling_rate=50.0, window_size=250, step_size=125)),
]
WINDOW_CONFIGS = [
    # ("2.0s", WindowConfig(sampling_rate=50.0, window_size=100, step_size=50)),
    ("2.5s", WindowConfig(sampling_rate=50.0, window_size=125, step_size=62)),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 TinyTCN rpm 细窗长扫描。")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--mode", choices=["full", "dev"], default="full")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    return parser.parse_args()


def write_summary_markdown(
    output_dir: Path, summary_df: pd.DataFrame, unlabeled_df: pd.DataFrame
) -> None:
    best_row = summary_df.iloc[0]
    lines = [
        "# TinyTCN rpm 细窗长扫描结论",
        "",
        f"- 最优窗长：`{best_row['window_label']}`",
        f"- 最优 case_mae：`{float(best_row['case_mae']):.4f}`",
        f"- 最优 case_rmse：`{float(best_row['case_rmse']):.4f}`",
        f"- 最优 case_mape：`{float(best_row['case_mape']):.4f}%`",
        "",
        "## 全部窗长",
        "",
    ]
    for _, row in summary_df.iterrows():
        lines.append(
            f"- `TinyTCN @ {row['window_label']}`: case_mae=`{float(row['case_mae']):.4f}`, case_rmse=`{float(row['case_rmse']):.4f}`, case_mape=`{float(row['case_mape']):.4f}%`, windows=`{int(row['window_count'])}`"
        )
    if not unlabeled_df.empty:
        lines.extend(["", "## 无标签工况推理", ""])
        for _, row in unlabeled_df.sort_values("window_label").iterrows():
            lines.append(
                f"- `{row['window_label']}`: `{row['file_name']}` 预测转速=`{float(row['predicted_rpm']):.4f} rpm`"
            )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


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
    train_config = TorchTrainConfig(
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
    )

    summary_rows: list[dict[str, object]] = []
    case_frames: list[pd.DataFrame] = []
    unlabeled_rows: list[pd.DataFrame] = []
    for window_label, window_config in WINDOW_CONFIGS:
        raw_dataset = build_raw_window_dataset(
            records, cleaned_signal_frames, window_config
        )
        prediction_frame = evaluate_tinytcn_rpm_loco(raw_dataset, train_config)
        summary_row, case_df = summarize_rpm_predictions(
            prediction_frame, f"TinyTCN@{window_label}"
        )
        summary_row["window_label"] = window_label
        summary_row["window_size"] = window_config.window_size
        summary_row["step_size"] = window_config.step_size
        summary_row["window_count"] = int(len(raw_dataset.meta_df))
        case_df["window_label"] = window_label
        case_frames.append(case_df)
        summary_rows.append(summary_row)

        unlabeled_df = predict_tinytcn_rpm_unlabeled(raw_dataset, train_config)
        if not unlabeled_df.empty:
            unlabeled_df["window_label"] = window_label
            unlabeled_rows.append(unlabeled_df)

    summary_df = (
        pd.DataFrame(summary_rows)
        .sort_values(["case_mae", "case_rmse", "window_label"])
        .reset_index(drop=True)
    )
    case_df = pd.concat(case_frames, ignore_index=True)
    unlabeled_df = (
        pd.concat(unlabeled_rows, ignore_index=True)
        if unlabeled_rows
        else pd.DataFrame()
    )

    summary_df.to_csv(
        output_dir / "rpm_fine_window_scan_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    case_df.to_csv(
        output_dir / "rpm_fine_window_scan_case_level_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    unlabeled_df.to_csv(
        output_dir / "rpm_fine_window_scan_unlabeled_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    write_summary_markdown(output_dir, summary_df, unlabeled_df)

    print("TinyTCN rpm 细窗长扫描已完成。")
    print(f"输出目录: {output_dir}")
    print(f"最优窗长: {summary_df.iloc[0]['window_label']}")


if __name__ == "__main__":
    main()
