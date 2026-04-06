from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

try:
    from fit_rpm import fit_selector_rpm, get_video_fps, load_annotations
    from task_io import resolve_jsonl_path, resolve_video_eval_path
except ModuleNotFoundError:
    from src.windNotFound.fit_rpm import fit_selector_rpm, get_video_fps, load_annotations
    from src.windNotFound.task_io import resolve_jsonl_path, resolve_video_eval_path

try:
    from src.windyWindHowfast.analysis_core import (
        analyze_spatiotemporal_frequency,
        build_angle_profile,
        extract_top_spectrum_peaks,
        select_preferred_peak,
    )
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from src.windyWindHowfast.analysis_core import (
        analyze_spatiotemporal_frequency,
        build_angle_profile,
        extract_top_spectrum_peaks,
        select_preferred_peak,
    )


def collect_selector_annotations(
    task: dict,
    annotations_by_task_item_id: dict[str, dict],
    selector_index: int,
) -> tuple[dict, list[dict]]:
    selectors = [selector for selector in task["selectors"] if selector["index"] == selector_index]
    if not selectors:
        raise ValueError(f"selector_index 不存在: {selector_index}")
    selector = selectors[0]
    selector_items = [item for item in task["items"] if item["selector_index"] == selector_index]
    records = [
        annotations_by_task_item_id[item["task_item_id"]]
        for item in selector_items
        if item["task_item_id"] in annotations_by_task_item_id
    ]
    dedup_by_frame: dict[int, dict] = {}
    for record in records:
        dedup_by_frame[int(record["frame_index"])] = record
    unique_records = [dedup_by_frame[frame_index] for frame_index in sorted(dedup_by_frame)]
    return selector, unique_records


def estimate_per_frame_roi(record: dict) -> tuple[tuple[int, int], int]:
    center = record["points"]["center"]
    blade_1 = record["points"]["blade_1"]
    radius = int(round(float(np.hypot(blade_1["x"] - center["x"], blade_1["y"] - center["y"]))))
    if radius < 8:
        raise ValueError("逐帧 ROI 半径过小，无法做频谱分析。")
    return (int(center["x"]), int(center["y"])), radius


def build_selector_time_angle_map(
    selector_records: list[dict],
    *,
    angular_res: int = 720,
    inner_radius_ratio: float = 0.25,
) -> tuple[np.ndarray, np.ndarray | None]:
    if len(selector_records) < 4:
        raise ValueError("至少需要 4 帧已标注记录才能做视频频谱分析。")

    cap = cv2.VideoCapture(selector_records[0]["video_path"])
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {selector_records[0]['video_path']}")

    try:
        time_angle_map = np.zeros((len(selector_records), angular_res), dtype=np.float32)
        debug_edges = None
        for idx, record in enumerate(selector_records):
            frame_index = int(record["frame_index"])
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError(f"无法读取指定帧: {frame_index}")
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            center, radius = estimate_per_frame_roi(record)
            angle_profile, edges = build_angle_profile(
                gray=gray,
                center=center,
                radius=radius,
                angular_res=angular_res,
                inner_radius_ratio=inner_radius_ratio,
            )
            time_angle_map[idx] = angle_profile
            if debug_edges is None:
                debug_edges = edges
    finally:
        cap.release()

    return time_angle_map, debug_edges


def evaluate_selector_video_rpm(
    task: dict,
    annotations_by_task_item_id: dict[str, dict],
    *,
    selector_index: int,
    angular_res: int = 720,
    inner_radius_ratio: float = 0.25,
    min_temporal_hz: float = 0.2,
    max_temporal_hz: float = 20.0,
    max_spatial_mode: int = 24,
) -> dict[str, object]:
    selector, records = collect_selector_annotations(task, annotations_by_task_item_id, selector_index)
    if not records:
        return {
            "selector_index": selector_index,
            "status": "no_annotations",
            "reason": "No annotation records found for this selector.",
        }

    fps = get_video_fps(records[0]["video_path"])
    manual_result = fit_selector_rpm(records, fps)
    try:
        time_angle_map, _ = build_selector_time_angle_map(
            records,
            angular_res=angular_res,
            inner_radius_ratio=inner_radius_ratio,
        )
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
        video_rpm = rotor_freq_hz * 60.0
        manual_rpm = manual_result["rpm"]
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
            raise RuntimeError("无法从频谱峰列表中选择有效峰。")
        selected_video_rpm = float(selected_peak["rpm"])
        return {
            "selector_index": selector_index,
            "selector_kind": selector["kind"],
            "source_id": selector["source"],
            "frame_indices": [record["frame_index"] for record in records],
            "annotated_item_count": len(records),
            "fps": fps,
            "status": "ok",
            "manual_fit": manual_result,
            "video_fft": {
                "rpm": selected_video_rpm,
                "raw_max_peak_rpm": video_rpm,
                "rotor_freq_hz": float(selected_video_rpm / 60.0),
                "peak_temporal_hz": float(selected_peak["temporal_hz"]),
                "peak_spatial_mode": int(selected_peak["spatial_mode_k"]),
                "peak_magnitude": float(selected_peak["magnitude"]),
                "selection_rule": "prefer_abs_k_eq_3_else_global_max",
                "raw_max_peak": {
                    "rpm": video_rpm,
                    "rotor_freq_hz": rotor_freq_hz,
                    "peak_temporal_hz": peak_temporal_hz,
                    "peak_spatial_mode": peak_spatial_mode,
                    "peak_magnitude": peak_magnitude,
                },
                "top_peaks": top_peaks,
            },
            "rpm_gap": None if manual_rpm is None else float(selected_video_rpm - manual_rpm),
            "abs_rpm_gap": None if manual_rpm is None else float(abs(selected_video_rpm - manual_rpm)),
        }
    except Exception as exc:
        return {
            "selector_index": selector_index,
            "selector_kind": selector["kind"],
            "source_id": selector["source"],
            "frame_indices": [record["frame_index"] for record in records],
            "annotated_item_count": len(records),
            "fps": fps,
            "status": "failed",
            "manual_fit": manual_result,
            "reason": str(exc),
        }


def evaluate_task_video_rpm(
    task: dict,
    *,
    selector_index: int,
    angular_res: int = 720,
    inner_radius_ratio: float = 0.25,
    min_temporal_hz: float = 0.2,
    max_temporal_hz: float = 20.0,
    max_spatial_mode: int = 24,
) -> dict[str, object]:
    jsonl_path = resolve_jsonl_path(task["task_path"])
    annotations_by_task_item_id, invalid_lines = load_annotations(jsonl_path)
    if invalid_lines:
        bad_lines = ", ".join(str(line_no) for line_no, _ in invalid_lines[:5])
        raise ValueError(f"jsonl contains invalid lines and cannot be used. Example line numbers: {bad_lines}")
    result = evaluate_selector_video_rpm(
        task,
        annotations_by_task_item_id,
        selector_index=selector_index,
        angular_res=angular_res,
        inner_radius_ratio=inner_radius_ratio,
        min_temporal_hz=min_temporal_hz,
        max_temporal_hz=max_temporal_hz,
        max_spatial_mode=max_spatial_mode,
    )
    payload = {
        "task_yaml": str(task["task_path"]),
        "task_name": task["task_path"].stem,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "summary_kind": "video_rpm_eval_with_per_frame_roi",
        "result": result,
    }
    output_path = resolve_video_eval_path(task["task_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "output_path": output_path,
        "payload": payload,
    }
