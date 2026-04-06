from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
TRY_NAME = "023_case5_51frame_anchor_5s_window_prediction"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "try" / TRY_NAME
DEFAULT_SUMMARY_JSON = REPO_ROOT / "outputs" / "annotations" / "test" / "summary.json"
DEFAULT_VIDEO_PATH = REPO_ROOT / "data" / "video" / "VID_20260330_162635.mp4"
DEFAULT_DATASET_INVENTORY = REPO_ROOT / "outputs" / "dataset_inventory.csv"
DEFAULT_WINDOW_PREDICTIONS = (
    REPO_ROOT
    / "outputs"
    / "try"
    / "022_case5_video_manual_label_validation"
    / "case5_loco_window_predictions.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行工况5 51帧锚点 5s 窗预测。")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY_JSON)
    parser.add_argument("--video-path", type=Path, default=DEFAULT_VIDEO_PATH)
    parser.add_argument("--dataset-inventory", type=Path, default=DEFAULT_DATASET_INVENTORY)
    parser.add_argument("--window-predictions", type=Path, default=DEFAULT_WINDOW_PREDICTIONS)
    return parser.parse_args()


def load_selector_info(summary_json: Path) -> dict[str, float]:
    payload = json.loads(summary_json.read_text(encoding="utf-8"))
    selector = payload["selectors"][0]
    frame_indices = selector["frame_indices"]
    center_frame = (min(frame_indices) + max(frame_indices)) / 2.0
    return {
        "frame_start": float(min(frame_indices)),
        "frame_end": float(max(frame_indices)),
        "frame_count": float(selector["annotated_item_count"]),
        "fps": float(selector["fps"]),
        "center_frame": center_frame,
        "center_time_sec": center_frame / float(selector["fps"]),
        "span_time_sec": float(selector["time_span_sec"]),
        "manual_rpm": float(selector["rpm"]),
    }


def load_video_info(video_path: Path) -> dict[str, float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return {
        "video_fps": fps,
        "video_frame_count": frame_count,
        "video_duration_sec": frame_count / fps if fps else 0.0,
    }


def load_case5_info(dataset_inventory_path: Path) -> dict[str, object]:
    inventory_df = pd.read_csv(dataset_inventory_path, encoding="utf-8-sig")
    row = inventory_df.loc[inventory_df["case_id"] == 5].iloc[0]
    return {
        "case_id": int(row["case_id"]),
        "file_name": row["file_name"],
        "table_start_time": row["start_time"],
        "table_end_time": row["end_time"],
        "table_duration_sec": float(row["duration_seconds"]),
    }


def build_alignment_summary(
    selector_info: dict[str, float],
    video_info: dict[str, float],
    case5_info: dict[str, object],
) -> pd.DataFrame:
    relative_position = selector_info["center_time_sec"] / video_info["video_duration_sec"]
    aligned_center_offset_sec = relative_position * float(case5_info["table_duration_sec"])
    return pd.DataFrame(
        [
            {
                "case_id": int(case5_info["case_id"]),
                "file_name": case5_info["file_name"],
                "video_duration_sec": float(video_info["video_duration_sec"]),
                "video_center_frame": float(selector_info["center_frame"]),
                "video_center_time_sec": float(selector_info["center_time_sec"]),
                "video_relative_position": float(relative_position),
                "table_duration_sec": float(case5_info["table_duration_sec"]),
                "aligned_center_offset_sec": float(aligned_center_offset_sec),
                "manual_rpm": float(selector_info["manual_rpm"]),
            }
        ]
    )


def select_aligned_window(
    window_predictions_path: Path,
    alignment_df: pd.DataFrame,
) -> pd.DataFrame:
    prediction_df = pd.read_csv(window_predictions_path, encoding="utf-8-sig")
    prediction_df["start_time"] = pd.to_datetime(prediction_df["start_time"])
    prediction_df["end_time"] = pd.to_datetime(prediction_df["end_time"])
    prediction_df["window_center_time"] = prediction_df["start_time"] + (
        prediction_df["end_time"] - prediction_df["start_time"]
    ) / 2
    case_start = prediction_df["start_time"].min()
    aligned_center_offset_sec = float(alignment_df.iloc[0]["aligned_center_offset_sec"])
    target_center_time = case_start + pd.to_timedelta(aligned_center_offset_sec, unit="s")
    prediction_df["target_center_time"] = target_center_time
    prediction_df["center_gap_sec"] = (
        prediction_df["window_center_time"] - target_center_time
    ).dt.total_seconds().abs()
    aligned_df = prediction_df.sort_values(["center_gap_sec", "window_index"]).head(1).copy()
    aligned_df["window_duration_sec"] = (
        aligned_df["end_time"] - aligned_df["start_time"]
    ).dt.total_seconds()
    return aligned_df


def write_summary(output_dir: Path, alignment_df: pd.DataFrame, aligned_df: pd.DataFrame) -> None:
    alignment_row = alignment_df.iloc[0]
    aligned_row = aligned_df.iloc[0]
    lines = [
        "# 工况5 51帧锚点 5s 窗预测结论",
        "",
        f"- 手工片段中心帧：`{float(alignment_row['video_center_frame']):.1f}`",
        f"- 手工片段中心时刻：`{float(alignment_row['video_center_time_sec']):.4f}s`",
        f"- 在整段视频中的相对位置：`{float(alignment_row['video_relative_position']) * 100:.2f}%`",
        f"- 映射到工况5表格后的中心偏移：`{float(alignment_row['aligned_center_offset_sec']):.4f}s`",
        "",
        "## 对齐到的 5s 窗",
        "",
        f"- 窗口索引：`{int(aligned_row['window_index'])}`",
        f"- 窗口时间：`{aligned_row['start_time']}` 到 `{aligned_row['end_time']}`",
        f"- 窗口中心：`{aligned_row['window_center_time']}`",
        f"- 与目标中心时刻差：`{float(aligned_row['center_gap_sec']):.4f}s`",
        f"- 该窗 TinyTCN 预测：`{float(aligned_row['pred_rpm']):.4f} rpm`",
        f"- 该窗 manifest 真值：`{float(aligned_row['true_rpm']):.4f} rpm`",
        f"- 手工 RPM：`{float(alignment_row['manual_rpm']):.4f} rpm`",
        "",
        "## 解释",
        "",
        "- 这是基于相对位置映射选出来的最接近 5s 窗，不是硬同步到 51 帧本身。",
        "- 如果后续拿到视频与表格的硬同步关系，应优先用硬同步替换这个近似映射。",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selector_info = load_selector_info(args.summary_json)
    video_info = load_video_info(args.video_path)
    case5_info = load_case5_info(args.dataset_inventory)
    alignment_df = build_alignment_summary(selector_info, video_info, case5_info)
    aligned_df = select_aligned_window(args.window_predictions, alignment_df)
    alignment_df.to_csv(args.output_dir / "alignment_summary.csv", index=False, encoding="utf-8-sig")
    aligned_df.to_csv(args.output_dir / "aligned_window_prediction.csv", index=False, encoding="utf-8-sig")
    write_summary(args.output_dir, alignment_df, aligned_df)
    print("工况5 51帧锚点 5s 窗预测已完成。")
    print(f"输出目录: {args.output_dir}")
    print(f"对齐窗口预测: {float(aligned_df.iloc[0]['pred_rpm']):.4f} rpm")


if __name__ == "__main__":
    main()
