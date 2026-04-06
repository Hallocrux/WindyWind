from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
TRY009_ROOT = REPO_ROOT / "src" / "try" / "009_phase1_feature_groups"
TRY012_ROOT = REPO_ROOT / "src" / "try" / "012_phase3_end_to_end_shortlist"
TRY019_ROOT = REPO_ROOT / "src" / "try" / "019_tinytcn_rpm_regression"
TRY021_ROOT = REPO_ROOT / "src" / "try" / "021_tinytcn_manual_labels_before_case5_validation"
for path in (REPO_ROOT, TRY009_ROOT, TRY012_ROOT, TRY019_ROOT, TRY021_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from src.current.data_loading import (
    DatasetRecord,
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

TRY_NAME = "021_tinytcn_manual_labels_before_case5_validation"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "try" / TRY_NAME
DEFAULT_SUMMARY_JSON = REPO_ROOT / "outputs" / "annotations" / "test" / "summary.json"
DEFAULT_VIDEO_EVAL_JSON = REPO_ROOT / "outputs" / "annotations" / "test" / "video_rpm_eval.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 TinyTCN 手工标注 pre-case5 验证。")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_ROOT,
        help="输出目录，默认写到 outputs/try/021_tinytcn_manual_labels_before_case5_validation。",
    )
    parser.add_argument(
        "--manual-summary-json",
        type=Path,
        default=DEFAULT_SUMMARY_JSON,
        help="手工标注 summary.json 路径。",
    )
    parser.add_argument(
        "--video-eval-json",
        type=Path,
        default=DEFAULT_VIDEO_EVAL_JSON,
        help="视频 RPM eval json 路径。",
    )
    parser.add_argument(
        "--max-case-id",
        type=int,
        default=4,
        help="只使用 case_id <= 该值的表格工况做 pre-case5 子验证。",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    return parser.parse_args()


def load_manual_reference(summary_json: Path, video_eval_json: Path) -> pd.DataFrame:
    summary_payload = json.loads(summary_json.read_text(encoding="utf-8"))
    video_eval_payload = json.loads(video_eval_json.read_text(encoding="utf-8"))
    selector = summary_payload["selectors"][0]
    result = video_eval_payload["result"]
    manual_fit = result["manual_fit"]
    video_fft = result["video_fft"]
    frame_indices = result["frame_indices"]
    row = {
        "task_name": summary_payload["task_name"],
        "task_yaml": summary_payload["task_yaml"],
        "video_path": selector["video_path"],
        "source_id": selector["source_id"],
        "selector_kind": selector["selector_kind"],
        "group": selector["group"],
        "role": selector["role"],
        "frame_start": min(frame_indices),
        "frame_end": max(frame_indices),
        "annotated_item_count": result["annotated_item_count"],
        "fps": result["fps"],
        "manual_rpm": manual_fit["rpm"],
        "manual_fit_residual_mae": manual_fit["fit_residual_mae"],
        "video_fft_rpm": video_fft["rpm"],
        "video_fft_abs_gap_to_manual": result["abs_rpm_gap"],
        "selection_rule": video_fft["selection_rule"],
    }
    return pd.DataFrame([row])


def select_pre_case5_records(max_case_id: int) -> list[DatasetRecord]:
    return [record for record in scan_dataset_records() if record.case_id <= max_case_id]


def build_candidate_case_frame(records: list[DatasetRecord], manual_rpm: float) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for record in records:
        rows.append(
            {
                "case_id": record.case_id,
                "display_name": record.display_name,
                "file_name": record.file_name,
                "label_source": record.label_source,
                "known_rpm": record.rpm,
                "is_labeled": record.rpm is not None,
                "abs_gap_to_manual_rpm": None
                if record.rpm is None
                else float(abs(record.rpm - manual_rpm)),
            }
        )
    candidate_df = pd.DataFrame(rows)
    return candidate_df.sort_values(
        ["is_labeled", "abs_gap_to_manual_rpm", "case_id"],
        ascending=[False, True, True],
        na_position="last",
    ).reset_index(drop=True)


def build_validation_summary(
    records: list[DatasetRecord],
    manual_reference_df: pd.DataFrame,
    loco_summary_row: dict[str, object],
    unlabeled_df: pd.DataFrame,
) -> pd.DataFrame:
    manual_row = manual_reference_df.iloc[0]
    predicted_rpm = None
    predicted_file = None
    gap_to_manual = None
    if not unlabeled_df.empty:
        pred_row = unlabeled_df.iloc[0]
        predicted_rpm = float(pred_row["predicted_rpm"])
        predicted_file = pred_row["file_name"]
        gap_to_manual = abs(predicted_rpm - float(manual_row["manual_rpm"]))
    labeled_case_ids = [record.case_id for record in records if record.rpm is not None]
    unlabeled_case_ids = [record.case_id for record in records if record.rpm is None]
    return pd.DataFrame(
        [
            {
                "max_case_id": max(record.case_id for record in records),
                "selected_case_ids": ",".join(str(record.case_id) for record in records),
                "labeled_case_ids": ",".join(str(case_id) for case_id in labeled_case_ids),
                "unlabeled_case_ids": ",".join(str(case_id) for case_id in unlabeled_case_ids),
                "manual_reference_task": manual_row["task_name"],
                "manual_reference_video": manual_row["video_path"],
                "manual_reference_rpm": float(manual_row["manual_rpm"]),
                "video_fft_rpm": float(manual_row["video_fft_rpm"]),
                "video_fft_abs_gap_to_manual": float(manual_row["video_fft_abs_gap_to_manual"]),
                "pre_case5_loco_case_mae": float(loco_summary_row["case_mae"]),
                "pre_case5_loco_case_rmse": float(loco_summary_row["case_rmse"]),
                "pre_case5_loco_case_mape": float(loco_summary_row["case_mape"]),
                "candidate_unlabeled_file": predicted_file,
                "candidate_predicted_rpm": predicted_rpm,
                "candidate_abs_gap_to_manual_rpm": gap_to_manual,
            }
        ]
    )


def write_summary_markdown(
    output_dir: Path,
    manual_reference_df: pd.DataFrame,
    candidate_case_df: pd.DataFrame,
    validation_df: pd.DataFrame,
) -> None:
    manual_row = manual_reference_df.iloc[0]
    validation_row = validation_df.iloc[0]
    top_labeled = candidate_case_df[candidate_case_df["is_labeled"]].iloc[0]
    lines = [
        "# TinyTCN 手工标注 pre-case5 验证结论",
        "",
        f"- 手工标注任务：`{manual_row['task_name']}`",
        f"- 手工 RPM：`{float(manual_row['manual_rpm']):.4f} rpm`",
        f"- 视频 FFT RPM：`{float(manual_row['video_fft_rpm']):.4f} rpm`",
        f"- 视频 FFT 与手工 RPM 的绝对差：`{float(manual_row['video_fft_abs_gap_to_manual']):.4f} rpm`",
        f"- pre-case5 参与工况：`{validation_row['selected_case_ids']}`",
        f"- pre-case5 LOCO case_mae：`{float(validation_row['pre_case5_loco_case_mae']):.4f} rpm`",
        f"- pre-case5 LOCO case_rmse：`{float(validation_row['pre_case5_loco_case_rmse']):.4f} rpm`",
        f"- pre-case5 LOCO case_mape：`{float(validation_row['pre_case5_loco_case_mape']):.4f}%`",
        "",
        "## 候选对比",
        "",
        f"- 与手工 RPM 最接近的已知 pre-case5 工况：`{top_labeled['file_name']}`，已知 rpm=`{float(top_labeled['known_rpm']):.4f}`，差值=`{float(top_labeled['abs_gap_to_manual_rpm']):.4f} rpm`",
    ]
    if pd.notna(validation_row["candidate_predicted_rpm"]):
        lines.append(
            f"- 无标签 `工况2` 的 TinyTCN 预测：`{float(validation_row['candidate_predicted_rpm']):.4f} rpm`，与手工 RPM 差值=`{float(validation_row['candidate_abs_gap_to_manual_rpm']):.4f} rpm`"
        )
    lines.extend(
        [
            "",
            "## 说明",
            "",
            "- 这个验证复用了仓库里已完成的手工标注 RPM 参考，但仓库内未找到该视频片段与表格工况编号的官方映射表。",
            "- 因此这里的结论应解读为：手工 RPM 参考与 pre-case5 TinyTCN 子验证是否处在一致量级，而不是严格工况级真值评测。",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    manual_reference_df = load_manual_reference(args.manual_summary_json, args.video_eval_json)
    manual_rpm = float(manual_reference_df.iloc[0]["manual_rpm"])
    records = select_pre_case5_records(args.max_case_id)
    if not records:
        raise ValueError("未找到任何 pre-case5 工况。")

    train_config = TorchTrainConfig(
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
    )
    common_signal_columns = get_common_signal_columns(records)
    cleaned_signal_frames = {
        record.case_id: load_clean_signal_frame(record, common_signal_columns)
        for record in records
    }
    raw_dataset = build_raw_window_dataset(records, cleaned_signal_frames, WindowConfig())

    prediction_frame = evaluate_tinytcn_rpm_loco(raw_dataset, train_config)
    loco_summary_row, case_df = summarize_rpm_predictions(prediction_frame, "TinyTCN")
    unlabeled_df = predict_tinytcn_rpm_unlabeled(raw_dataset, train_config)
    candidate_case_df = build_candidate_case_frame(records, manual_rpm)
    validation_df = build_validation_summary(records, manual_reference_df, loco_summary_row, unlabeled_df)

    manual_reference_df.to_csv(
        output_dir / "manual_reference_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    candidate_case_df.to_csv(
        output_dir / "pre_case5_candidate_case_rpm.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame([loco_summary_row]).to_csv(
        output_dir / "pre_case5_loco_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    case_df.to_csv(
        output_dir / "pre_case5_case_level_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    unlabeled_df.to_csv(
        output_dir / "pre_case5_unlabeled_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    validation_df.to_csv(
        output_dir / "validation_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    write_summary_markdown(output_dir, manual_reference_df, candidate_case_df, validation_df)

    print("TinyTCN 手工标注 pre-case5 验证已完成。")
    print(f"输出目录: {output_dir}")
    print(f"手工 RPM: {manual_rpm:.4f} rpm")
    if not unlabeled_df.empty:
        first_row = unlabeled_df.iloc[0]
        print(f"{first_row['file_name']} 预测转速: {float(first_row['predicted_rpm']):.4f} rpm")


if __name__ == "__main__":
    main()
