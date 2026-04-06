from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

try:
    from task_io import resolve_jsonl_path, resolve_summary_path
except ModuleNotFoundError:
    from src.windNotFound.task_io import resolve_jsonl_path, resolve_summary_path


def load_annotations(jsonl_path: str | Path) -> tuple[dict[str, dict], list[tuple[int, str]]]:
    jsonl_path = Path(jsonl_path).resolve()
    by_task_item_id: dict[str, dict] = {}
    invalid_lines: list[tuple[int, str]] = []
    if not jsonl_path.exists():
        return by_task_item_id, invalid_lines
    with jsonl_path.open("r", encoding="utf-8") as fp:
        for line_number, raw_line in enumerate(fp, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
                by_task_item_id[str(payload["task_item_id"])] = payload
            except Exception as exc:
                invalid_lines.append((line_number, str(exc)))
    return by_task_item_id, invalid_lines


def get_video_fps(video_path: str) -> float:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频以读取 FPS: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    if fps <= 0:
        raise RuntimeError(f"视频 FPS 非法: {video_path}")
    return fps


def fit_selector_rpm(records: list[dict], fps: float) -> dict[str, object]:
    if fps <= 0:
        return {
            "status": "invalid_fps",
            "rpm": None,
            "signed_rpm": None,
            "angular_velocity_rad_per_sec": None,
            "frame_span": None,
            "time_span_sec": None,
            "fit_residual_mae": None,
            "reason": "Video FPS must be positive.",
        }

    unique_by_frame: dict[int, dict] = {}
    for record in records:
        unique_by_frame[int(record["frame_index"])] = record
    unique_records = [unique_by_frame[k] for k in sorted(unique_by_frame)]
    if len(unique_records) < 2:
        return {
            "status": "insufficient_annotations",
            "rpm": None,
            "signed_rpm": None,
            "angular_velocity_rad_per_sec": None,
            "frame_span": None,
            "time_span_sec": None,
            "fit_residual_mae": None,
            "reason": "At least two unique annotated frames are required.",
        }

    frame_indices = np.asarray([item["frame_index"] for item in unique_records], dtype=np.float64)
    times = frame_indices / fps
    if np.allclose(times, times[0]):
        return {
            "status": "invalid_time_span",
            "rpm": None,
            "signed_rpm": None,
            "angular_velocity_rad_per_sec": None,
            "frame_span": 0,
            "time_span_sec": 0.0,
            "fit_residual_mae": None,
            "reason": "Annotated frames collapse to zero duration.",
        }

    angles = []
    for record in unique_records:
        center = record["points"]["center"]
        blade_1 = record["points"]["blade_1"]
        dy = blade_1["y"] - center["y"]
        dx = blade_1["x"] - center["x"]
        angles.append(float(np.arctan2(dy, dx)))
    unwrapped = np.unwrap(np.asarray(angles, dtype=np.float64))
    slope, intercept = np.polyfit(times, unwrapped, deg=1)
    predicted = slope * times + intercept
    residual_mae = float(np.mean(np.abs(unwrapped - predicted)))
    signed_rpm = float(slope * 60.0 / (2.0 * np.pi))
    return {
        "status": "ok",
        "rpm": abs(signed_rpm),
        "signed_rpm": signed_rpm,
        "angular_velocity_rad_per_sec": float(slope),
        "frame_span": int(frame_indices[-1] - frame_indices[0]),
        "time_span_sec": float(times[-1] - times[0]),
        "fit_residual_mae": residual_mae,
        "reason": None,
    }


def build_summary_payload(
    task: dict,
    annotations_by_task_item_id: dict[str, dict],
    jsonl_path: str | Path,
) -> dict[str, object]:
    selectors_payload: list[dict[str, object]] = []
    for selector in task["selectors"]:
        selector_items = [item for item in task["items"] if item["selector_index"] == selector["index"]]
        selector_annotations = [
            annotations_by_task_item_id[item["task_item_id"]]
            for item in selector_items
            if item["task_item_id"] in annotations_by_task_item_id
        ]
        selector_annotations.sort(key=lambda item: (item["frame_index"], item["task_item_id"]))
        fps = get_video_fps(task["sources"][selector["source"]]["video"])
        selectors_payload.append(
            {
                "selector_index": selector["index"],
                "selector_kind": selector["kind"],
                "source_id": selector["source"],
                "video_path": task["sources"][selector["source"]]["video"],
                "group": selector["group"],
                "role": selector["role"],
                "note": selector["note"],
                "planned_item_count": len(selector_items),
                "annotated_item_count": len(selector_annotations),
                "frame_indices": [item["frame_index"] for item in selector_annotations],
                "fps": fps,
                **fit_selector_rpm(selector_annotations, fps),
            }
        )
    return {
        "task_yaml": str(task["task_path"]),
        "task_name": task["task_path"].stem,
        "task_version": task["version"],
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "jsonl_path": str(Path(jsonl_path).resolve()),
        "selector_count": len(task["selectors"]),
        "summary_kind": "annotation_fit_rpm",
        "selectors": selectors_payload,
    }


def write_summary(task: dict, annotations_by_task_item_id: dict[str, dict], summary_path: str | Path) -> Path:
    path = Path(summary_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_summary_payload(task, annotations_by_task_item_id, resolve_jsonl_path(task["task_path"]))
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def fit_task_rpm(task: dict) -> dict[str, object]:
    jsonl_path = resolve_jsonl_path(task["task_path"])
    annotations_by_task_item_id, invalid_lines = load_annotations(jsonl_path)
    if invalid_lines:
        bad_lines = ", ".join(str(line_no) for line_no, _ in invalid_lines[:5])
        raise ValueError(f"jsonl contains invalid lines and cannot be used. Example line numbers: {bad_lines}")
    summary_path = write_summary(task, annotations_by_task_item_id, resolve_summary_path(task["task_path"]))
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    return {
        "summary_path": summary_path,
        "payload": payload,
    }
