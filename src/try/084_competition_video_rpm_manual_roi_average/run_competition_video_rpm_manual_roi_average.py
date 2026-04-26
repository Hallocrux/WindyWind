from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.windNotFound.fit_rpm import load_annotations
from src.windNotFound.task_io import load_task, resolve_jsonl_path  # noqa: E402
from src.windNotFound.video_rpm_eval import evaluate_selector_video_rpm  # noqa: E402
from src.windyWindHowfast.analysis_core import (  # noqa: E402
    analyze_spatiotemporal_frequency,
    build_angle_profile,
    extract_top_spectrum_peaks,
    select_preferred_peak,
)

DEFAULT_OUTPUT_DIR = Path("outputs/try/084_competition_video_rpm_manual_roi_average")
DEFAULT_SEGMENT_COUNT = 3
DEFAULT_FRAMES_PER_SEGMENT = 51
DEFAULT_SELECTOR_STEP = 1
DEFAULT_MAX_INTERNAL_GAP_ROWS = 5
DEFAULT_INNER_RADIUS_RATIO = 0.25
DEFAULT_ANGULAR_RES = 720
DEFAULT_MIN_TEMPORAL_HZ = 0.2
DEFAULT_MAX_TEMPORAL_HZ = 20.0
DEFAULT_MAX_SPATIAL_MODE = 24
INVALID_COLUMNS = {
    "WSMS00005.AccX",
    "WSMS00005.AccY",
    "WSMS00005.AccZ",
}
VIDEO_END_TIME_PATTERN = re.compile(r"(?P<date>\d{8})_(?P<time>\d{6})")
ROI_WINDOW_NAME = "Competition Video ROI Annotation"


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    fps: float
    frame_count: int
    duration_sec: float
    start_time: datetime
    end_time: datetime


@dataclass(frozen=True)
class SegmentWindow:
    selector_index: int
    center_time: datetime
    center_frame: int
    start_frame: int
    end_frame: int
    start_time: datetime
    end_time: datetime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="竞赛测试视频多片段手工 ROI RPM 平均工具。")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare", help="生成多片段标注任务与预览。")
    prepare_parser.add_argument("--video", type=Path, required=True, help="视频路径。")
    prepare_parser.add_argument("--csv", type=Path, required=True, help="对应 CSV 路径。")
    prepare_parser.add_argument(
        "--video-end-time",
        type=str,
        default=None,
        help='显式指定视频结束时间，例如 "2026-04-09 13:48:05"。',
    )
    prepare_parser.add_argument("--segment-count", type=int, default=DEFAULT_SEGMENT_COUNT, help="中段片段数。")
    prepare_parser.add_argument(
        "--frames-per-segment",
        type=int,
        default=DEFAULT_FRAMES_PER_SEGMENT,
        help="每个片段的帧数，建议为奇数。",
    )
    prepare_parser.add_argument("--selector-step", type=int, default=DEFAULT_SELECTOR_STEP, help="selector 采样步长。")
    prepare_parser.add_argument(
        "--max-internal-gap-rows",
        type=int,
        default=DEFAULT_MAX_INTERNAL_GAP_ROWS,
        help="CSV 清洗时允许保留的中间短缺失最大长度。",
    )
    prepare_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="输出目录。")

    annotate_parser = subparsers.add_parser("annotate-roi", help="对每个 selector 的中心帧手工标注静态 ROI。")
    annotate_parser.add_argument("--task", type=Path, required=True, help="prepare 生成的任务 YAML 路径。")
    annotate_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="结果输出目录。")

    aggregate_parser = subparsers.add_parser("aggregate", help="汇总已有标注的多片段 RPM。")
    aggregate_parser.add_argument("--task", type=Path, required=True, help="prepare 生成的任务 YAML 路径。")
    aggregate_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="结果输出目录。")
    aggregate_parser.add_argument("--angular-res", type=int, default=DEFAULT_ANGULAR_RES, help="角度分辨率。")
    aggregate_parser.add_argument(
        "--inner-radius-ratio",
        type=float,
        default=DEFAULT_INNER_RADIUS_RATIO,
        help="忽略中心区域的半径比例。",
    )
    aggregate_parser.add_argument("--min-temporal-hz", type=float, default=DEFAULT_MIN_TEMPORAL_HZ, help="时间频率下界。")
    aggregate_parser.add_argument("--max-temporal-hz", type=float, default=DEFAULT_MAX_TEMPORAL_HZ, help="时间频率上界。")
    aggregate_parser.add_argument("--max-spatial-mode", type=int, default=DEFAULT_MAX_SPATIAL_MODE, help="空间模态上界。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        run_prepare(args)
        return
    if args.command == "annotate-roi":
        run_annotate_roi(args)
        return
    if args.command == "aggregate":
        run_aggregate(args)
        return
    raise ValueError(f"未知命令: {args.command}")


def run_prepare(args: argparse.Namespace) -> None:
    output_dir: Path = args.output_dir.resolve()
    task_dir = output_dir / "tasks"
    preview_dir = output_dir / "previews"
    result_dir = output_dir / "results"
    task_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    video_path = args.video.resolve()
    csv_path = args.csv.resolve()
    video_end_time = parse_video_end_time(video_path, args.video_end_time)
    video_info = load_video_info(video_path, video_end_time)
    cleaned_csv = load_clean_csv_frame(
        csv_path=csv_path,
        max_internal_gap_rows=int(args.max_internal_gap_rows),
    )
    overlap_summary = compute_best_overlap_segment(cleaned_csv, video_info)
    segment_windows = build_segment_windows(
        overlap_start=overlap_summary["overlap_start"],
        overlap_end=overlap_summary["overlap_end"],
        video_info=video_info,
        segment_count=int(args.segment_count),
        frames_per_segment=int(args.frames_per_segment),
        selector_step=int(args.selector_step),
    )

    task_stem = f"competition_video_{video_path.stem}_manual_roi_average"
    task_path = task_dir / f"{task_stem}.yaml"
    context_path = result_dir / f"{task_stem}_context.json"
    summary_path = result_dir / f"{task_stem}_prepare_summary.md"
    write_task_yaml(
        task_path=task_path,
        video_path=video_path,
        segment_windows=segment_windows,
        selector_step=int(args.selector_step),
    )
    export_preview_frames(
        video_path=video_path,
        preview_dir=preview_dir / task_stem,
        segment_windows=segment_windows,
    )
    context_payload = build_prepare_context_payload(
        video_info=video_info,
        csv_path=csv_path,
        cleaned_csv=cleaned_csv,
        overlap_summary=overlap_summary,
        segment_windows=segment_windows,
        task_path=task_path,
        preview_dir=(preview_dir / task_stem),
    )
    context_path.write_text(json.dumps(context_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_path.write_text(build_prepare_summary_markdown(context_payload), encoding="utf-8")

    print(f"任务文件: {task_path}")
    print(f"上下文文件: {context_path}")
    print(f"预览目录: {preview_dir / task_stem}")
    print(f"说明文件: {summary_path}")
    print(f"ROI 标注文件: {resolve_selector_roi_path(task_path, output_dir)}")


def run_annotate_roi(args: argparse.Namespace) -> None:
    task_path = args.task.resolve()
    output_dir: Path = args.output_dir.resolve()
    task = load_task(task_path)
    selector_roi_path = resolve_selector_roi_path(task_path, output_dir)
    roi_annotations = annotate_selector_rois(task, selector_roi_path)
    selector_roi_path.write_text(json.dumps(roi_annotations, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"ROI 标注文件: {selector_roi_path}")


def run_aggregate(args: argparse.Namespace) -> None:
    output_dir: Path = args.output_dir.resolve()
    result_dir = output_dir / "results"
    result_dir.mkdir(parents=True, exist_ok=True)

    task_path = args.task.resolve()
    task = load_task(task_path)
    selector_roi_path = resolve_selector_roi_path(task_path, output_dir)
    selector_roi_payload = load_selector_roi_payload(selector_roi_path)
    selector_results: list[dict[str, Any]]
    aggregation_mode: str
    if selector_roi_payload.get("selectors"):
        selector_results = []
        for selector in task["selectors"]:
            roi_record = selector_roi_payload["selectors"].get(str(int(selector["index"])))
            if roi_record is None:
                selector_results.append(
                    {
                        "selector_index": int(selector["index"]),
                        "status": "no_roi_annotation",
                        "reason": "No static ROI annotation found for this selector.",
                    }
                )
                continue
            selector_results.append(
                evaluate_selector_with_static_roi(
                    task=task,
                    selector_index=int(selector["index"]),
                    roi_record=roi_record,
                    angular_res=int(args.angular_res),
                    inner_radius_ratio=float(args.inner_radius_ratio),
                    min_temporal_hz=float(args.min_temporal_hz),
                    max_temporal_hz=float(args.max_temporal_hz),
                    max_spatial_mode=int(args.max_spatial_mode),
                )
            )
        aggregation_mode = "static_selector_roi"
    else:
        jsonl_path = resolve_jsonl_path(task_path)
        annotations_by_task_item_id, invalid_lines = load_annotations(jsonl_path)
        if invalid_lines:
            bad_lines = ", ".join(str(line_no) for line_no, _ in invalid_lines[:5])
            raise ValueError(f"标注文件存在非法行，示例行号: {bad_lines}")
        selector_results = []
        for selector in task["selectors"]:
            result = evaluate_selector_video_rpm(
                task,
                annotations_by_task_item_id,
                selector_index=int(selector["index"]),
                angular_res=int(args.angular_res),
                inner_radius_ratio=float(args.inner_radius_ratio),
                min_temporal_hz=float(args.min_temporal_hz),
                max_temporal_hz=float(args.max_temporal_hz),
                max_spatial_mode=int(args.max_spatial_mode),
            )
            selector_results.append(result)
        aggregation_mode = "wind_not_found_per_frame_roi"

    aggregate_payload = build_aggregate_payload(task_path, selector_results, aggregation_mode)
    task_stem = task_path.stem
    json_path = result_dir / f"{task_stem}_aggregate.json"
    md_path = result_dir / f"{task_stem}_aggregate.md"
    json_path.write_text(json.dumps(aggregate_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(build_aggregate_markdown(aggregate_payload), encoding="utf-8")

    print(f"聚合结果 JSON: {json_path}")
    print(f"聚合结果 Markdown: {md_path}")
    print(f"成功片段数: {aggregate_payload['successful_selector_count']}")
    print(f"最终平均 RPM: {aggregate_payload['final_average_rpm']}")


def parse_video_end_time(video_path: Path, explicit_value: str | None) -> datetime:
    if explicit_value:
        return pd.Timestamp(explicit_value).to_pydatetime()
    matched = VIDEO_END_TIME_PATTERN.search(video_path.stem)
    if not matched:
        raise ValueError(
            "无法从视频文件名解析结束时间，请显式传入 --video-end-time。"
        )
    text = matched.group("date") + matched.group("time")
    return datetime.strptime(text, "%Y%m%d%H%M%S")


def load_video_info(video_path: Path, end_time: datetime) -> VideoInfo:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(round(float(cap.get(cv2.CAP_PROP_FRAME_COUNT))))
    cap.release()
    if fps <= 0 or frame_count <= 0:
        raise RuntimeError(f"视频元数据非法: fps={fps}, frame_count={frame_count}")
    duration_sec = frame_count / fps
    start_time = end_time - timedelta(seconds=duration_sec)
    return VideoInfo(
        path=video_path,
        fps=fps,
        frame_count=frame_count,
        duration_sec=duration_sec,
        start_time=start_time,
        end_time=end_time,
    )


def load_clean_csv_frame(csv_path: Path, max_internal_gap_rows: int) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)
    if "time" not in frame.columns:
        raise ValueError(f"{csv_path} 缺少 time 列。")

    signal_columns = [column for column in frame.columns if column not in {"time", *INVALID_COLUMNS}]
    if not signal_columns:
        raise ValueError(f"{csv_path} 没有可用信号列。")

    cleaned_time = clean_time_series(frame["time"])
    cleaned = frame.loc[:, ["time", *signal_columns]].copy()
    cleaned["time"] = pd.to_datetime(cleaned_time, errors="coerce")
    cleaned = cleaned.dropna(subset=["time"]).sort_values("time").drop_duplicates(subset="time", keep="first")
    cleaned = cleaned.reset_index(drop=True)

    numeric = cleaned[signal_columns].apply(pd.to_numeric, errors="coerce")
    row_has_missing = numeric.isna().sum(axis=1).gt(0)
    leading_mask, trailing_mask = edge_missing_masks(row_has_missing.to_numpy(dtype=bool, copy=False))
    keep_edge_mask = ~(leading_mask | trailing_mask)
    cleaned = cleaned.loc[keep_edge_mask].reset_index(drop=True)
    numeric = numeric.loc[keep_edge_mask].reset_index(drop=True)
    if cleaned.empty:
        raise ValueError(f"{csv_path} 删除首尾连续缺失段后无有效数据。")

    row_has_missing = numeric.isna().sum(axis=1).gt(0)
    missing_blocks = collect_missing_blocks(row_has_missing.to_numpy(dtype=bool, copy=False))
    long_gap_mask = np.zeros(len(cleaned), dtype=bool)
    for block in missing_blocks:
        if block["length"] > max_internal_gap_rows:
            long_gap_mask[block["start"] : block["end"] + 1] = True

    keep_middle_mask = ~long_gap_mask
    source_indices = np.flatnonzero(keep_middle_mask)
    cleaned = cleaned.loc[keep_middle_mask].reset_index(drop=True)
    if cleaned.empty:
        raise ValueError(f"{csv_path} 删除中间长缺失段后无有效数据。")
    cleaned["__segment_id"] = build_segment_ids(source_indices)
    return cleaned


def compute_best_overlap_segment(cleaned_csv: pd.DataFrame, video_info: VideoInfo) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for segment_id, segment_df in cleaned_csv.groupby("__segment_id", sort=True):
        segment_df = segment_df.reset_index(drop=True)
        segment_start = segment_df["time"].iloc[0].to_pydatetime()
        segment_end = segment_df["time"].iloc[-1].to_pydatetime()
        overlap_start = max(segment_start, video_info.start_time)
        overlap_end = min(segment_end, video_info.end_time)
        overlap_sec = max(0.0, (overlap_end - overlap_start).total_seconds())
        candidate = {
            "segment_id": int(segment_id),
            "segment_start": segment_start,
            "segment_end": segment_end,
            "segment_rows": int(len(segment_df)),
            "overlap_start": overlap_start,
            "overlap_end": overlap_end,
            "overlap_sec": overlap_sec,
        }
        if overlap_sec <= 0:
            continue
        if best is None or candidate["overlap_sec"] > best["overlap_sec"]:
            best = candidate

    if best is None:
        raise ValueError("视频时间范围与 CSV 清洗后的连续稳定段没有重叠。")
    return best


def build_segment_windows(
    *,
    overlap_start: datetime,
    overlap_end: datetime,
    video_info: VideoInfo,
    segment_count: int,
    frames_per_segment: int,
    selector_step: int,
) -> list[SegmentWindow]:
    if segment_count <= 0:
        raise ValueError("segment-count 必须为正整数。")
    if frames_per_segment < 5:
        raise ValueError("frames-per-segment 至少应为 5。")
    if frames_per_segment % 2 == 0:
        raise ValueError("frames-per-segment 必须为奇数，便于围绕中心帧对称取片段。")
    if selector_step <= 0:
        raise ValueError("selector-step 必须为正整数。")

    half_span_frames = (frames_per_segment - 1) // 2 * selector_step
    half_span_sec = half_span_frames / video_info.fps
    overlap_duration_sec = (overlap_end - overlap_start).total_seconds()
    if overlap_duration_sec <= 2 * half_span_sec:
        raise ValueError("视频与 CSV 的稳定重叠区过短，无法放下完整的人工标注片段。")

    center_margin_sec = max(half_span_sec + 0.25, overlap_duration_sec * 0.1)
    center_start = overlap_start + timedelta(seconds=center_margin_sec)
    center_end = overlap_end - timedelta(seconds=center_margin_sec)
    if center_end <= center_start:
        center_start = overlap_start + timedelta(seconds=half_span_sec)
        center_end = overlap_end - timedelta(seconds=half_span_sec)
    if center_end <= center_start:
        raise ValueError("稳定重叠区不足以生成多个中段片段。")

    if segment_count == 1:
        center_times = [center_start + (center_end - center_start) / 2]
    else:
        total_span_sec = (center_end - center_start).total_seconds()
        center_times = [
            center_start + timedelta(seconds=total_span_sec * idx / (segment_count - 1))
            for idx in range(segment_count)
        ]

    windows: list[SegmentWindow] = []
    for selector_index, center_time in enumerate(center_times):
        center_frame = timestamp_to_video_frame(center_time, video_info)
        start_frame = center_frame - half_span_frames
        end_frame = center_frame + half_span_frames
        if start_frame < 0 or end_frame >= video_info.frame_count:
            raise ValueError("生成的片段越界，请缩短片段长度或检查时间对齐。")
        start_time = video_info.start_time + timedelta(seconds=start_frame / video_info.fps)
        end_time = video_info.start_time + timedelta(seconds=end_frame / video_info.fps)
        windows.append(
            SegmentWindow(
                selector_index=selector_index,
                center_time=center_time,
                center_frame=center_frame,
                start_frame=start_frame,
                end_frame=end_frame,
                start_time=start_time,
                end_time=end_time,
            )
        )
    return windows


def timestamp_to_video_frame(timestamp: datetime, video_info: VideoInfo) -> int:
    seconds = (timestamp - video_info.start_time).total_seconds()
    frame = int(round(seconds * video_info.fps))
    return min(max(frame, 0), video_info.frame_count - 1)


def write_task_yaml(
    *,
    task_path: Path,
    video_path: Path,
    segment_windows: list[SegmentWindow],
    selector_step: int,
) -> None:
    before = (segment_windows[0].center_frame - segment_windows[0].start_frame) // selector_step
    after = (segment_windows[0].end_frame - segment_windows[0].center_frame) // selector_step
    payload = {
        "version": 1,
        "sources": [
            {
                "id": "competition_video",
                "video": str(video_path),
            }
        ],
        "selectors": [],
    }
    for window in segment_windows:
        payload["selectors"].append(
            {
                "kind": "window",
                "source": "competition_video",
                "center": int(window.center_frame),
                "before": int(before),
                "after": int(after),
                "step": int(selector_step),
                "role": "middle_manual_roi",
                "group": f"segment_{window.selector_index + 1:02d}",
                "note": (
                    f"aligned_time={window.center_time.isoformat(sep=' ', timespec='milliseconds')}"
                ),
            }
        )
    task_path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")


def export_preview_frames(
    *,
    video_path: Path,
    preview_dir: Path,
    segment_windows: list[SegmentWindow],
) -> None:
    preview_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频生成预览: {video_path}")
    try:
        for window in segment_windows:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(window.center_frame))
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError(f"无法读取预览帧: {window.center_frame}")
            output_path = preview_dir / f"selector_{window.selector_index:02d}_frame_{window.center_frame}.png"
            cv2.imwrite(str(output_path), frame)
    finally:
        cap.release()


def resolve_selector_roi_path(task_path: Path, output_dir: Path) -> Path:
    return output_dir.resolve() / "results" / f"{task_path.stem}_selector_rois.json"


def load_selector_roi_payload(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"selectors": {}}
    payload = json.loads(path.read_text(encoding="utf-8"))
    selectors = payload.get("selectors")
    if not isinstance(selectors, dict):
        return {"selectors": {}}
    return payload


def annotate_selector_rois(task: dict[str, Any], selector_roi_path: Path) -> dict[str, Any]:
    existing_payload = load_selector_roi_payload(selector_roi_path)
    selector_records = {
        str(key): value
        for key, value in existing_payload.get("selectors", {}).items()
        if isinstance(value, dict)
    }
    ordered_selectors = sorted(task["selectors"], key=lambda item: int(item["index"]))
    current_index = 0
    points: list[tuple[int, int]] = []
    scale = 1.0
    frame_cache: dict[int, np.ndarray] = {}

    def load_selector_frame(selector_index: int) -> np.ndarray:
        if selector_index in frame_cache:
            return frame_cache[selector_index].copy()
        selector = next(item for item in ordered_selectors if int(item["index"]) == selector_index)
        center_frame = int(selector["center"])
        video_path = task["sources"][selector["source"]]["video"]
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频: {video_path}")
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, center_frame)
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError(f"无法读取 selector 中心帧: {center_frame}")
            frame_cache[selector_index] = frame.copy()
            return frame
        finally:
            cap.release()

    def load_existing_points(selector_index: int) -> list[tuple[int, int]]:
        record = selector_records.get(str(selector_index))
        if not record:
            return []
        center = record.get("center")
        edge = record.get("radius_point")
        if not isinstance(center, dict) or not isinstance(edge, dict):
            return []
        return [
            (int(center["x"]), int(center["y"])),
            (int(edge["x"]), int(edge["y"])),
        ]

    def draw_canvas(selector_index: int, current_points: list[tuple[int, int]]) -> np.ndarray:
        selector = next(item for item in ordered_selectors if int(item["index"]) == selector_index)
        frame = load_selector_frame(selector_index).copy()
        info_lines = [
            f"Selector {selector_index + 1}/{len(ordered_selectors)} | frame={selector['center']} | group={selector.get('group') or '-'}",
            f"note={selector.get('note') or '-'}",
            "Step 1: click rotor center | Step 2: click blade tip to define radius",
            "Keys: z undo | n next if current selector already saved | p previous | q/Esc quit",
        ]
        for idx, line in enumerate(info_lines):
            y = 28 + idx * 24
            cv2.putText(frame, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(frame, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30, 30, 30), 1, cv2.LINE_AA)

        if current_points:
            center = current_points[0]
            cv2.circle(frame, center, 5, (0, 0, 255), -1)
            cv2.putText(frame, "center", (center[0] + 8, center[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)
        if len(current_points) >= 2:
            radius_point = current_points[1]
            cv2.circle(frame, radius_point, 5, (0, 255, 0), -1)
            cv2.putText(frame, "radius", (radius_point[0] + 8, radius_point[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)
            radius = int(round(math.hypot(radius_point[0] - center[0], radius_point[1] - center[1])))
            cv2.circle(frame, center, radius, (0, 255, 255), 2)
            cv2.line(frame, center, radius_point, (0, 255, 255), 2)

        frame_height = frame.shape[0]
        nonlocal_scale = 900 / frame_height if frame_height > 900 else 1.0
        nonlocal scale
        scale = nonlocal_scale
        if scale != 1.0:
            frame = cv2.resize(frame, None, fx=scale, fy=scale)
        return frame

    def save_selector(selector_index: int, current_points: list[tuple[int, int]]) -> None:
        center = current_points[0]
        radius_point = current_points[1]
        selector = next(item for item in ordered_selectors if int(item["index"]) == selector_index)
        radius = int(round(math.hypot(radius_point[0] - center[0], radius_point[1] - center[1])))
        selector_records[str(selector_index)] = {
            "selector_index": int(selector_index),
            "source_id": selector["source"],
            "center_frame": int(selector["center"]),
            "center": {"x": int(center[0]), "y": int(center[1])},
            "radius_point": {"x": int(radius_point[0]), "y": int(radius_point[1])},
            "radius": int(radius),
            "annotated_at": datetime.now().isoformat(timespec="seconds"),
        }
        selector_roi_path.parent.mkdir(parents=True, exist_ok=True)
        selector_roi_path.write_text(
            json.dumps(
                {
                    "task_yaml": str(task["task_path"]),
                    "generated_at": datetime.now().isoformat(timespec="seconds"),
                    "selectors": selector_records,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def on_click(event: int, x: int, y: int, flags: int, param: object) -> None:
        _ = flags, param
        if event != cv2.EVENT_LBUTTONDOWN or scale <= 0:
            return
        real_x = int(round(x / scale))
        real_y = int(round(y / scale))
        if len(points) >= 2:
            return
        points.append((real_x, real_y))
        if len(points) == 2:
            save_selector(int(ordered_selectors[current_index]["index"]), points)

    cv2.namedWindow(ROI_WINDOW_NAME)
    cv2.setMouseCallback(ROI_WINDOW_NAME, on_click)
    try:
        while True:
            selector_index = int(ordered_selectors[current_index]["index"])
            existing_points = load_existing_points(selector_index)
            if not points and existing_points:
                points[:] = existing_points
            canvas = draw_canvas(selector_index, points)
            cv2.imshow(ROI_WINDOW_NAME, canvas)
            key = cv2.waitKey(20) & 0xFF
            if key == 255:
                continue
            if key in {27, ord("q")}:
                break
            if key == ord("z"):
                if points:
                    points.pop()
                elif existing_points:
                    points[:] = existing_points[:-1]
                continue
            if key == ord("n"):
                if str(selector_index) in selector_records and current_index < len(ordered_selectors) - 1:
                    current_index += 1
                    points = []
                continue
            if key == ord("p"):
                if current_index > 0:
                    current_index -= 1
                    points = []
                continue
            if len(points) == 2 and current_index < len(ordered_selectors) - 1:
                current_index += 1
                points = []
    finally:
        cv2.destroyAllWindows()

    return {
        "task_yaml": str(task["task_path"]),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "selectors": selector_records,
    }


def collect_selector_frame_indices(task: dict[str, Any], selector_index: int) -> tuple[dict[str, Any], list[int]]:
    selectors = [selector for selector in task["selectors"] if int(selector["index"]) == int(selector_index)]
    if not selectors:
        raise ValueError(f"selector_index 不存在: {selector_index}")
    selector = selectors[0]
    frame_indices = sorted(
        {
            int(item["frame_index"])
            for item in task["items"]
            if int(item["selector_index"]) == int(selector_index)
        }
    )
    return selector, frame_indices


def evaluate_selector_with_static_roi(
    *,
    task: dict[str, Any],
    selector_index: int,
    roi_record: dict[str, Any],
    angular_res: int,
    inner_radius_ratio: float,
    min_temporal_hz: float,
    max_temporal_hz: float,
    max_spatial_mode: int,
) -> dict[str, Any]:
    selector, frame_indices = collect_selector_frame_indices(task, selector_index)
    if len(frame_indices) < 4:
        return {
            "selector_index": int(selector_index),
            "status": "insufficient_frames",
            "reason": "Selector frame count is below 4.",
        }

    center = (int(roi_record["center"]["x"]), int(roi_record["center"]["y"]))
    radius = int(roi_record["radius"])
    video_path = task["sources"][selector["source"]]["video"]
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        cap.release()
        raise RuntimeError(f"视频 FPS 非法: {video_path}")

    time_angle_map = np.zeros((len(frame_indices), angular_res), dtype=np.float32)
    try:
        for index, frame_index in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError(f"无法读取指定帧: {frame_index}")
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            angle_profile, _ = build_angle_profile(
                gray=gray,
                center=center,
                radius=radius,
                angular_res=angular_res,
                inner_radius_ratio=inner_radius_ratio,
            )
            time_angle_map[index] = angle_profile
    finally:
        cap.release()

    (
        rotor_freq_hz,
        peak_temporal_hz,
        peak_spatial_mode,
        peak_magnitude,
        _processed_map,
        spectrum_mag,
        temporal_freqs,
        spatial_modes,
    ) = analyze_spatiotemporal_frequency(
        time_angle_map=time_angle_map,
        fps=fps,
        min_temporal_hz=min_temporal_hz,
        max_temporal_hz=max_temporal_hz,
        max_spatial_mode=max_spatial_mode,
    )
    raw_max_peak_rpm = rotor_freq_hz * 60.0
    top_peaks = extract_top_spectrum_peaks(
        spectrum_mag=spectrum_mag,
        temporal_freqs=temporal_freqs,
        spatial_modes=spatial_modes,
        min_temporal_hz=min_temporal_hz,
        max_temporal_hz=max_temporal_hz,
        max_spatial_mode=max_spatial_mode,
        top_n=10,
    )
    selected_peak = select_preferred_peak(top_peaks, preferred_abs_k=(3,))
    if selected_peak is None:
        return {
            "selector_index": int(selector_index),
            "status": "failed",
            "reason": "无法从频谱峰列表中选择有效峰。",
        }

    return {
        "selector_index": int(selector_index),
        "selector_kind": selector["kind"],
        "source_id": selector["source"],
        "frame_indices": frame_indices,
        "annotated_item_count": len(frame_indices),
        "fps": fps,
        "status": "ok",
        "annotation_mode": "static_selector_roi",
        "selector_roi": {
            "center_x": int(center[0]),
            "center_y": int(center[1]),
            "radius": int(radius),
        },
        "manual_fit": None,
        "video_fft": {
            "rpm": float(selected_peak["rpm"]),
            "raw_max_peak_rpm": float(raw_max_peak_rpm),
            "rotor_freq_hz": float(selected_peak["rpm"] / 60.0),
            "peak_temporal_hz": float(selected_peak["temporal_hz"]),
            "peak_spatial_mode": int(selected_peak["spatial_mode_k"]),
            "peak_magnitude": float(selected_peak["magnitude"]),
            "selection_rule": "prefer_abs_k_eq_3_else_global_max",
            "raw_max_peak": {
                "rpm": float(raw_max_peak_rpm),
                "rotor_freq_hz": float(rotor_freq_hz),
                "peak_temporal_hz": float(peak_temporal_hz),
                "peak_spatial_mode": int(peak_spatial_mode),
                "peak_magnitude": float(peak_magnitude),
            },
            "top_peaks": top_peaks,
        },
        "reason": None,
    }


def build_prepare_context_payload(
    *,
    video_info: VideoInfo,
    csv_path: Path,
    cleaned_csv: pd.DataFrame,
    overlap_summary: dict[str, Any],
    segment_windows: list[SegmentWindow],
    task_path: Path,
    preview_dir: Path,
) -> dict[str, Any]:
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "video": {
            "path": str(video_info.path),
            "fps": video_info.fps,
            "frame_count": video_info.frame_count,
            "duration_sec": video_info.duration_sec,
            "start_time": video_info.start_time.isoformat(sep=" ", timespec="milliseconds"),
            "end_time": video_info.end_time.isoformat(sep=" ", timespec="seconds"),
        },
        "csv": {
            "path": str(csv_path),
            "rows_after_cleaning": int(len(cleaned_csv)),
            "time_start": cleaned_csv["time"].iloc[0].isoformat(sep=" ", timespec="milliseconds"),
            "time_end": cleaned_csv["time"].iloc[-1].isoformat(sep=" ", timespec="milliseconds"),
            "segment_count": int(cleaned_csv["__segment_id"].nunique()),
        },
        "selected_overlap_segment": {
            "segment_id": int(overlap_summary["segment_id"]),
            "segment_start": overlap_summary["segment_start"].isoformat(sep=" ", timespec="milliseconds"),
            "segment_end": overlap_summary["segment_end"].isoformat(sep=" ", timespec="milliseconds"),
            "segment_rows": int(overlap_summary["segment_rows"]),
            "overlap_start": overlap_summary["overlap_start"].isoformat(sep=" ", timespec="milliseconds"),
            "overlap_end": overlap_summary["overlap_end"].isoformat(sep=" ", timespec="milliseconds"),
            "overlap_sec": float(overlap_summary["overlap_sec"]),
        },
        "task_yaml": str(task_path),
        "preview_dir": str(preview_dir),
        "selectors": [
            {
                "selector_index": int(window.selector_index),
                "center_frame": int(window.center_frame),
                "start_frame": int(window.start_frame),
                "end_frame": int(window.end_frame),
                "center_time": window.center_time.isoformat(sep=" ", timespec="milliseconds"),
                "start_time": window.start_time.isoformat(sep=" ", timespec="milliseconds"),
                "end_time": window.end_time.isoformat(sep=" ", timespec="milliseconds"),
            }
            for window in segment_windows
        ],
    }


def build_prepare_summary_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# 竞赛测试视频多片段手工 ROI 任务",
        "",
        "## 时间对齐",
        "",
        f"- 视频：`{payload['video']['path']}`",
        f"- 视频开始时间：`{payload['video']['start_time']}`",
        f"- 视频结束时间：`{payload['video']['end_time']}`",
        f"- CSV：`{payload['csv']['path']}`",
        f"- CSV 清洗后时间范围：`{payload['csv']['time_start']}` 到 `{payload['csv']['time_end']}`",
        f"- 选中的重叠稳定段：`{payload['selected_overlap_segment']['overlap_start']}` 到 `{payload['selected_overlap_segment']['overlap_end']}`",
        f"- 重叠时长：`{payload['selected_overlap_segment']['overlap_sec']:.3f}s`",
        "",
        "## 标注任务",
        "",
        f"- 任务文件：`{payload['task_yaml']}`",
        f"- 预览目录：`{payload['preview_dir']}`",
        "",
        "## 片段列表",
        "",
    ]
    for selector in payload["selectors"]:
        lines.append(
            f"- selector {selector['selector_index']}: frame `{selector['start_frame']}-{selector['end_frame']}` | "
            f"time `{selector['start_time']}` 到 `{selector['end_time']}`"
        )
    lines.extend(
        [
            "",
            "## 下一步",
            "",
            "- 先用本探索的 `annotate-roi` 子命令完成每个 selector 的静态 ROI 标注。",
            "- 标注完成后，再运行本探索的 `aggregate` 子命令输出各片段 RPM 和最终平均值。",
        ]
    )
    return "\n".join(lines)


def build_aggregate_payload(
    task_path: Path,
    selector_results: list[dict[str, Any]],
    aggregation_mode: str,
) -> dict[str, Any]:
    successful = [item for item in selector_results if item.get("status") == "ok"]
    video_rpms = [float(item["video_fft"]["rpm"]) for item in successful]
    manual_rpms = [
        float(item["manual_fit"]["rpm"])
        for item in successful
        if isinstance(item.get("manual_fit"), dict) and item["manual_fit"].get("rpm") is not None
    ]
    selector_payloads: list[dict[str, Any]] = []
    for item in selector_results:
        manual_fit = item.get("manual_fit")
        video_fft = item.get("video_fft")
        selector_payloads.append(
            {
                "selector_index": int(item["selector_index"]),
                "status": item["status"],
                "frame_indices": item.get("frame_indices", []),
                "annotated_item_count": item.get("annotated_item_count"),
                "manual_fit_rpm": None
                if not isinstance(manual_fit, dict) or manual_fit.get("rpm") is None
                else float(manual_fit["rpm"]),
                "video_fft_rpm": None
                if not isinstance(video_fft, dict) or video_fft.get("rpm") is None
                else float(video_fft["rpm"]),
                "selected_spatial_mode": None
                if not isinstance(video_fft, dict) or video_fft.get("peak_spatial_mode") is None
                else int(video_fft["peak_spatial_mode"]),
                "abs_rpm_gap": None if item.get("abs_rpm_gap") is None else float(item["abs_rpm_gap"]),
                "reason": item.get("reason"),
            }
        )

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "task_yaml": str(task_path),
        "aggregation_mode": aggregation_mode,
        "selector_count": len(selector_results),
        "successful_selector_count": len(successful),
        "failed_selector_count": len(selector_results) - len(successful),
        "final_average_rpm": None if not video_rpms else float(np.mean(video_rpms)),
        "final_rpm_std": None if len(video_rpms) < 2 else float(np.std(video_rpms, ddof=0)),
        "manual_average_rpm": None if not manual_rpms else float(np.mean(manual_rpms)),
        "manual_rpm_std": None if len(manual_rpms) < 2 else float(np.std(manual_rpms, ddof=0)),
        "selector_results": selector_payloads,
    }


def build_aggregate_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# 多片段 RPM 聚合结果",
        "",
        f"- task：`{payload['task_yaml']}`",
        f"- 聚合模式：`{payload['aggregation_mode']}`",
        f"- selector 总数：`{payload['selector_count']}`",
        f"- 成功 selector 数：`{payload['successful_selector_count']}`",
        f"- 失败 selector 数：`{payload['failed_selector_count']}`",
        f"- 最终平均 RPM：`{format_optional_float(payload['final_average_rpm'])}`",
        f"- 最终 RPM 标准差：`{format_optional_float(payload['final_rpm_std'])}`",
        f"- 人工角度拟合平均 RPM：`{format_optional_float(payload['manual_average_rpm'])}`",
        f"- 人工角度拟合标准差：`{format_optional_float(payload['manual_rpm_std'])}`",
        "",
        "## 各片段结果",
        "",
    ]
    for item in payload["selector_results"]:
        lines.append(
            f"- selector {item['selector_index']}: status=`{item['status']}` | "
            f"video_fft_rpm=`{format_optional_float(item['video_fft_rpm'])}` | "
            f"manual_fit_rpm=`{format_optional_float(item['manual_fit_rpm'])}` | "
            f"abs_gap=`{format_optional_float(item['abs_rpm_gap'])}`"
        )
        if item["reason"]:
            lines.append(f"- selector {item['selector_index']} reason: `{item['reason']}`")
    return "\n".join(lines)


def clean_time_series(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.strip()
    text = text.str.removeprefix('="').str.removeprefix("=")
    return text.str.strip('"')


def edge_missing_masks(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    leading = np.zeros(len(mask), dtype=bool)
    trailing = np.zeros(len(mask), dtype=bool)

    index = 0
    while index < len(mask) and mask[index]:
        leading[index] = True
        index += 1

    index = len(mask) - 1
    while index >= 0 and mask[index]:
        trailing[index] = True
        index -= 1
    return leading, trailing


def collect_missing_blocks(mask: np.ndarray) -> list[dict[str, int]]:
    blocks: list[dict[str, int]] = []
    start: int | None = None
    for index, is_missing in enumerate(mask):
        if is_missing and start is None:
            start = index
        elif not is_missing and start is not None:
            blocks.append({"start": start, "end": index - 1, "length": index - start})
            start = None
    if start is not None:
        blocks.append({"start": start, "end": len(mask) - 1, "length": len(mask) - start})
    return blocks


def build_segment_ids(source_indices: np.ndarray) -> np.ndarray:
    if len(source_indices) == 0:
        return np.array([], dtype=int)
    segment_ids = np.zeros(len(source_indices), dtype=int)
    if len(source_indices) == 1:
        return segment_ids
    segment_ids[1:] = np.cumsum(np.diff(source_indices) > 1)
    return segment_ids


def format_optional_float(value: float | None) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "None"
    return f"{value:.6f}"


if __name__ == "__main__":
    main()
