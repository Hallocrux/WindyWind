from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
TRY009_ROOT = REPO_ROOT / "src" / "try" / "009_phase1_feature_groups"
TRY012_ROOT = REPO_ROOT / "src" / "try" / "012_phase3_end_to_end_shortlist"
TRY019_ROOT = REPO_ROOT / "src" / "try" / "019_tinytcn_rpm_regression"
TRY022_ROOT = REPO_ROOT / "src" / "try" / "022_case5_video_manual_label_validation"
for path in (REPO_ROOT, TRY009_ROOT, TRY012_ROOT, TRY019_ROOT, TRY022_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from src.current.data_loading import (  # noqa: E402
    get_common_signal_columns,
    load_clean_signal_frame,
    scan_dataset_records,
)
from src.current.features import WindowConfig  # noqa: E402

from phase3_end_to_end_lib import build_raw_window_dataset  # noqa: E402
from tinytcn_rpm_lib import (  # noqa: E402
    TinyTCN,
    TorchTrainConfig,
    normalize_windows_by_channel,
    train_torch_model,
)

TRY_NAME = "022_case5_video_manual_label_validation"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "try" / TRY_NAME
DEFAULT_SUMMARY_JSON = REPO_ROOT / "outputs" / "annotations" / "test" / "summary.json"
DEFAULT_VIDEO_EVAL_JSON = REPO_ROOT / "outputs" / "annotations" / "test" / "video_rpm_eval.json"
DEFAULT_DATASET_INVENTORY = REPO_ROOT / "outputs" / "dataset_inventory.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行工况5视频手工标注定向验证。")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_ROOT,
        help="输出目录，默认写到 outputs/try/022_case5_video_manual_label_validation。",
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
        "--dataset-inventory",
        type=Path,
        default=DEFAULT_DATASET_INVENTORY,
        help="数据清单 csv 路径。",
    )
    parser.add_argument("--case-id", type=int, default=5, help="默认验证工况编号。")
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
    row = {
        "task_name": summary_payload["task_name"],
        "task_yaml": summary_payload["task_yaml"],
        "video_path": selector["video_path"],
        "source_id": selector["source_id"],
        "selector_kind": selector["selector_kind"],
        "group": selector["group"],
        "role": selector["role"],
        "frame_start": min(result["frame_indices"]),
        "frame_end": max(result["frame_indices"]),
        "annotated_item_count": result["annotated_item_count"],
        "fps": result["fps"],
        "manual_rpm": manual_fit["rpm"],
        "manual_fit_residual_mae": manual_fit["fit_residual_mae"],
        "video_fft_rpm": video_fft["rpm"],
        "video_fft_abs_gap_to_manual": result["abs_rpm_gap"],
        "selection_rule": video_fft["selection_rule"],
    }
    return pd.DataFrame([row])


def build_case5_evidence(case_id: int, dataset_inventory_path: Path) -> pd.DataFrame:
    records = scan_dataset_records()
    case_record = next(record for record in records if record.case_id == case_id)
    inventory_df = pd.read_csv(dataset_inventory_path, encoding="utf-8-sig")
    inventory_row = inventory_df.loc[inventory_df["case_id"] == case_id].iloc[0]
    return pd.DataFrame(
        [
            {
                "case_id": case_record.case_id,
                "display_name": case_record.display_name,
                "file_name": case_record.file_name,
                "manifest_rpm": case_record.rpm,
                "manifest_wind_speed": case_record.wind_speed,
                "label_source": case_record.label_source,
                "notes": case_record.notes,
                "table_start_time": inventory_row["start_time"],
                "table_end_time": inventory_row["end_time"],
                "table_duration_seconds": float(inventory_row["duration_seconds"]),
                "sampling_hz_est": float(inventory_row["sampling_hz_est"]),
            }
        ]
    )


def run_case_holdout_prediction(case_id: int, config: TorchTrainConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    records = scan_dataset_records()
    common_signal_columns = get_common_signal_columns(records)
    cleaned_signal_frames = {
        record.case_id: load_clean_signal_frame(record, common_signal_columns)
        for record in records
    }
    raw_dataset = build_raw_window_dataset(records, cleaned_signal_frames, WindowConfig())
    labeled_mask = raw_dataset.meta_df["rpm"].notna().to_numpy()
    labeled_meta = raw_dataset.meta_df.loc[labeled_mask].reset_index(drop=True)
    labeled_windows = raw_dataset.windows[labeled_mask]
    case_values = labeled_meta["case_id"].to_numpy(dtype=int, copy=False)
    train_idx = case_values != case_id
    valid_idx = case_values == case_id
    if not valid_idx.any():
        raise ValueError(f"未找到 case_id={case_id} 的带标签窗口。")

    X_train = labeled_windows[train_idx]
    X_valid = labeled_windows[valid_idx]
    y_train = labeled_meta.loc[train_idx, "rpm"].to_numpy(dtype="float32", copy=False)
    y_valid = labeled_meta.loc[valid_idx, "rpm"].to_numpy(dtype="float32", copy=False)

    X_train_norm, X_valid_norm = normalize_windows_by_channel(X_train, X_valid)
    torch.manual_seed(42)
    model = TinyTCN(in_channels=X_train.shape[1]).to(torch.device("cpu"))
    train_torch_model(
        model=model,
        X_train=X_train_norm,
        y_train=y_train,
        X_valid=X_valid_norm,
        y_valid=y_valid,
        config=config,
        device=torch.device("cpu"),
    )
    with torch.no_grad():
        pred = model(torch.from_numpy(X_valid_norm)).cpu().numpy()

    valid_df = labeled_meta.loc[valid_idx, ["case_id", "file_name", "window_index", "start_time", "end_time", "rpm"]].copy()
    valid_df = valid_df.rename(columns={"rpm": "true_rpm"})
    valid_df["pred_rpm"] = pred
    case_df = (
        valid_df.groupby(["case_id", "file_name", "true_rpm"], as_index=False)["pred_rpm"]
        .mean()
        .rename(columns={"pred_rpm": "pred_mean"})
    )
    case_df["abs_error_vs_manifest"] = (case_df["pred_mean"] - case_df["true_rpm"]).abs()
    return valid_df, case_df


def build_comparison_summary(
    manual_reference_df: pd.DataFrame,
    case5_evidence_df: pd.DataFrame,
    case5_case_df: pd.DataFrame,
) -> pd.DataFrame:
    manual_row = manual_reference_df.iloc[0]
    case5_row = case5_evidence_df.iloc[0]
    pred_row = case5_case_df.iloc[0]
    manual_rpm = float(manual_row["manual_rpm"])
    video_fft_rpm = float(manual_row["video_fft_rpm"])
    manifest_rpm = float(case5_row["manifest_rpm"])
    predicted_rpm = float(pred_row["pred_mean"])
    return pd.DataFrame(
        [
            {
                "case_id": int(case5_row["case_id"]),
                "file_name": case5_row["file_name"],
                "manual_rpm": manual_rpm,
                "video_fft_rpm": video_fft_rpm,
                "manifest_rpm": manifest_rpm,
                "tinytcn_case5_holdout_pred_rpm": predicted_rpm,
                "manual_vs_video_fft_abs_gap": abs(manual_rpm - video_fft_rpm),
                "manual_vs_manifest_abs_gap": abs(manual_rpm - manifest_rpm),
                "manual_vs_tinytcn_abs_gap": abs(manual_rpm - predicted_rpm),
                "video_fft_vs_manifest_abs_gap": abs(video_fft_rpm - manifest_rpm),
                "video_fft_vs_tinytcn_abs_gap": abs(video_fft_rpm - predicted_rpm),
                "manifest_vs_tinytcn_abs_gap": abs(manifest_rpm - predicted_rpm),
            }
        ]
    )


def build_anchor_window_summary(
    manual_reference_df: pd.DataFrame,
    case5_evidence_df: pd.DataFrame,
    case5_window_df: pd.DataFrame,
) -> pd.DataFrame:
    manual_row = manual_reference_df.iloc[0]
    case5_row = case5_evidence_df.iloc[0]
    fps = float(manual_row["fps"])
    frame_start = int(manual_row["frame_start"])
    frame_end = int(manual_row["frame_end"])
    center_frame = (frame_start + frame_end) / 2.0
    anchor_offset_sec = center_frame / fps
    table_start = pd.Timestamp(case5_row["table_start_time"])
    anchor_timestamp = table_start + pd.to_timedelta(anchor_offset_sec, unit="s")

    window_df = case5_window_df.copy()
    window_df["start_time"] = pd.to_datetime(window_df["start_time"])
    window_df["end_time"] = pd.to_datetime(window_df["end_time"])
    window_df["window_center_time"] = window_df["start_time"] + (window_df["end_time"] - window_df["start_time"]) / 2
    window_df["anchor_abs_gap_sec"] = (
        (window_df["window_center_time"] - anchor_timestamp).abs().dt.total_seconds()
    )
    best_row = window_df.sort_values(["anchor_abs_gap_sec", "window_index"]).iloc[0]

    return pd.DataFrame(
        [
            {
                "case_id": int(case5_row["case_id"]),
                "file_name": case5_row["file_name"],
                "alignment_assumption": "video_start_aligned_to_case5_table_start",
                "video_fps": fps,
                "video_frame_start": frame_start,
                "video_frame_end": frame_end,
                "video_frame_count": frame_end - frame_start + 1,
                "video_center_frame": center_frame,
                "video_center_offset_sec": anchor_offset_sec,
                "table_start_time": str(table_start),
                "anchor_timestamp": str(anchor_timestamp),
                "selected_window_index": int(best_row["window_index"]),
                "selected_window_start_time": str(best_row["start_time"]),
                "selected_window_end_time": str(best_row["end_time"]),
                "selected_window_center_time": str(best_row["window_center_time"]),
                "selected_window_pred_rpm": float(best_row["pred_rpm"]),
                "selected_window_true_rpm": float(best_row["true_rpm"]),
                "selected_window_anchor_abs_gap_sec": float(best_row["anchor_abs_gap_sec"]),
            }
        ]
    )


def write_summary_markdown(
    output_dir: Path,
    manual_reference_df: pd.DataFrame,
    case5_evidence_df: pd.DataFrame,
    case5_case_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    anchor_window_df: pd.DataFrame,
) -> None:
    manual_row = manual_reference_df.iloc[0]
    case5_row = case5_evidence_df.iloc[0]
    pred_row = case5_case_df.iloc[0]
    comparison_row = comparison_df.iloc[0]
    anchor_row = anchor_window_df.iloc[0]
    lines = [
        "# 工况5视频手工标注定向验证结论",
        "",
        f"- 验证工况：`工况{int(case5_row['case_id'])}` / `{case5_row['file_name']}`",
        f"- 视频：`{manual_row['video_path']}`",
        f"- 手工标注帧范围：`{int(manual_row['frame_start'])}-{int(manual_row['frame_end'])}`",
        f"- 手工标注帧数：`{int(anchor_row['video_frame_count'])}`",
        f"- 手工 RPM：`{float(manual_row['manual_rpm']):.4f} rpm`",
        f"- 视频 FFT RPM：`{float(manual_row['video_fft_rpm']):.4f} rpm`",
        f"- manifest RPM：`{float(case5_row['manifest_rpm']):.4f} rpm`",
        f"- TinyTCN 工况5 holdout 预测：`{float(pred_row['pred_mean']):.4f} rpm`",
        "",
        "## 差值对比",
        "",
        f"- 手工 vs 视频 FFT：`{float(comparison_row['manual_vs_video_fft_abs_gap']):.4f} rpm`",
        f"- 手工 vs manifest：`{float(comparison_row['manual_vs_manifest_abs_gap']):.4f} rpm`",
        f"- 手工 vs TinyTCN：`{float(comparison_row['manual_vs_tinytcn_abs_gap']):.4f} rpm`",
        f"- manifest vs TinyTCN：`{float(comparison_row['manifest_vs_tinytcn_abs_gap']):.4f} rpm`",
        "",
        "## 51帧中心时刻对齐的现有 5s 窗",
        "",
        f"- 工作假设：`{anchor_row['alignment_assumption']}`",
        f"- 片段中心帧：`{float(anchor_row['video_center_frame']):.1f}`，对应视频内偏移 `约 {float(anchor_row['video_center_offset_sec']):.4f}s`",
        f"- 锚点表格时刻：`{anchor_row['anchor_timestamp']}`",
        f"- 最接近的现有 `5s` 窗：`#{int(anchor_row['selected_window_index'])}`，`{anchor_row['selected_window_start_time']}` 到 `{anchor_row['selected_window_end_time']}`",
        f"- 该窗预测 RPM：`{float(anchor_row['selected_window_pred_rpm']):.4f}`，与锚点绝对时差 `约 {float(anchor_row['selected_window_anchor_abs_gap_sec']):.4f}s`",
        "",
        "## 解释",
        "",
        "- 这个 try 直接围绕 `工况5` 做定向验证，不再把工况5视频误写成 pre-case5 外部参考。",
        "- 当前 `5s` 窗预测只是把 51 帧中心时刻映射到最接近的现有表格窗，不等于“模型直接对 51 帧片段预测”。",
        "- 如果手工 RPM 与 manifest / TinyTCN 持续存在明显差值，更可能说明视频片段只覆盖了工况5内部某个局部阶段，或视频侧与表格侧的同步关系还需要继续细化。",
    ]
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

    manual_reference_df = load_manual_reference(args.manual_summary_json, args.video_eval_json)
    case5_evidence_df = build_case5_evidence(args.case_id, args.dataset_inventory)
    case5_window_df, case5_case_df = run_case_holdout_prediction(args.case_id, train_config)
    comparison_df = build_comparison_summary(manual_reference_df, case5_evidence_df, case5_case_df)
    anchor_window_df = build_anchor_window_summary(manual_reference_df, case5_evidence_df, case5_window_df)

    manual_reference_df.to_csv(output_dir / "manual_reference_summary.csv", index=False, encoding="utf-8-sig")
    case5_evidence_df.to_csv(output_dir / "case5_evidence_summary.csv", index=False, encoding="utf-8-sig")
    case5_window_df.to_csv(output_dir / "case5_loco_window_predictions.csv", index=False, encoding="utf-8-sig")
    comparison_df.to_csv(output_dir / "case5_comparison_summary.csv", index=False, encoding="utf-8-sig")
    anchor_window_df.to_csv(output_dir / "case5_anchor_5s_window_prediction.csv", index=False, encoding="utf-8-sig")
    write_summary_markdown(
        output_dir,
        manual_reference_df,
        case5_evidence_df,
        case5_case_df,
        comparison_df,
        anchor_window_df,
    )

    print("工况5视频手工标注定向验证已完成。")
    print(f"输出目录: {output_dir}")
    print(f"工况5 TinyTCN holdout 预测: {float(case5_case_df.iloc[0]['pred_mean']):.4f} rpm")


if __name__ == "__main__":
    main()
