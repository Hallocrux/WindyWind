from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import yaml


ANNOTATION_POINT_ORDER = [
    "support_a",
    "support_b",
    "center",
    "blade_1",
    "blade_2",
    "blade_3",
]

POINT_INSTRUCTIONS = {
    "support_a": "Step 1: click the first support-pole point.",
    "support_b": "Step 2: click the second support-pole point.",
    "center": "Step 3: click the rotor center.",
    "blade_1": "Step 4: click the marker blade edge as blade_1.",
    "blade_2": "Step 5: click the next blade edge clockwise as blade_2.",
    "blade_3": "Step 6: click the last blade edge clockwise as blade_3.",
}

TASK_CONFIG_VERSION = 1


def resolve_output_dir(task_path: str | Path) -> Path:
    return Path("outputs").resolve() / "annotations" / Path(task_path).resolve().stem


def resolve_jsonl_path(task_path: str | Path) -> Path:
    return resolve_output_dir(task_path) / "annotations.jsonl"


def resolve_summary_path(task_path: str | Path) -> Path:
    return resolve_output_dir(task_path) / "summary.json"


def resolve_video_eval_path(task_path: str | Path) -> Path:
    return resolve_output_dir(task_path) / "video_rpm_eval.json"


def _ensure_mapping(data: Any, context: str) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError(f"{context} 必须是对象。")
    return data


def _ensure_int(data: dict[str, Any], key: str, context: str) -> int:
    value = data.get(key)
    if not isinstance(value, int):
        raise ValueError(f"{context}.{key} 必须是整数。")
    return value


def _ensure_optional_str(data: dict[str, Any], key: str) -> str | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{key} 必须是字符串。")
    return value


def _resolve_video_path(task_path: Path, video_value: str) -> str:
    candidate = Path(video_value)
    if candidate.is_absolute():
        return str(candidate)
    local_candidate = (task_path.parent / candidate).resolve()
    if local_candidate.exists():
        return str(local_candidate)
    return str(candidate.resolve())


def _expand_selector_frames(selector: dict[str, Any]) -> list[int]:
    kind = selector["kind"]
    if kind == "window":
        center = selector["center"]
        before = selector["before"]
        after = selector["after"]
        step = selector["step"]
        return list(range(center - before * step, center + after * step + 1, step))
    if kind == "range":
        start = selector["start"]
        end = selector["end"]
        step = selector["step"]
        if end < start:
            raise ValueError(f"selector[{selector['index']}] 的 end 不能小于 start。")
        return list(range(start, end + 1, step))
    if kind == "explicit":
        return sorted(selector["frames"])
    raise ValueError(f"不支持的 selector kind: {kind}")


def load_task(task_path: str | Path) -> dict[str, Any]:
    task_path = Path(task_path).resolve()
    payload = _ensure_mapping(yaml.safe_load(task_path.read_text(encoding="utf-8")), "task")
    version = payload.get("version")
    if version != TASK_CONFIG_VERSION:
        raise ValueError(f"仅支持 version={TASK_CONFIG_VERSION}，当前为 {version!r}。")

    raw_sources = payload.get("sources")
    if not isinstance(raw_sources, list) or not raw_sources:
        raise ValueError("sources 必须是非空列表。")
    sources: dict[str, dict[str, Any]] = {}
    for idx, item in enumerate(raw_sources):
        source_data = _ensure_mapping(item, f"sources[{idx}]")
        source_id = source_data.get("id")
        video = source_data.get("video")
        if not isinstance(source_id, str) or not source_id:
            raise ValueError(f"sources[{idx}].id 必须是非空字符串。")
        if not isinstance(video, str) or not video:
            raise ValueError(f"sources[{idx}].video 必须是非空字符串。")
        sources[source_id] = {
            "id": source_id,
            "video": _resolve_video_path(task_path, video),
        }

    raw_selectors = payload.get("selectors")
    if not isinstance(raw_selectors, list) or not raw_selectors:
        raise ValueError("selectors 必须是非空列表。")

    selectors: list[dict[str, Any]] = []
    items: list[dict[str, Any]] = []
    for selector_index, item in enumerate(raw_selectors):
        selector_data = _ensure_mapping(item, f"selectors[{selector_index}]")
        kind = selector_data.get("kind")
        source_id = selector_data.get("source")
        if kind not in {"window", "range", "explicit"}:
            raise ValueError(f"selectors[{selector_index}].kind 不支持: {kind!r}")
        if not isinstance(source_id, str) or source_id not in sources:
            raise ValueError(f"selectors[{selector_index}].source 无效: {source_id!r}")
        step = selector_data.get("step", 1)
        if not isinstance(step, int) or step <= 0:
            raise ValueError(f"selectors[{selector_index}].step 必须是正整数。")

        selector = {
            "index": selector_index,
            "kind": kind,
            "source": source_id,
            "role": _ensure_optional_str(selector_data, "role"),
            "group": _ensure_optional_str(selector_data, "group"),
            "note": _ensure_optional_str(selector_data, "note"),
            "center": _ensure_int(selector_data, "center", f"selectors[{selector_index}]") if kind == "window" else None,
            "before": _ensure_int(selector_data, "before", f"selectors[{selector_index}]") if kind == "window" else None,
            "after": _ensure_int(selector_data, "after", f"selectors[{selector_index}]") if kind == "window" else None,
            "start": _ensure_int(selector_data, "start", f"selectors[{selector_index}]") if kind == "range" else None,
            "end": _ensure_int(selector_data, "end", f"selectors[{selector_index}]") if kind == "range" else None,
            "step": step,
            "frames": tuple(sorted({int(frame) for frame in selector_data.get("frames", [])})) if kind == "explicit" else (),
        }
        if kind == "explicit" and not selector["frames"]:
            raise ValueError(f"selectors[{selector_index}].frames 不能为空。")
        selectors.append(selector)

        for frame_index in _expand_selector_frames(selector):
            if frame_index < 0:
                raise ValueError(f"selectors[{selector_index}] 生成了负 frame: {frame_index}")
            items.append(
                {
                    "task_item_id": f"{selector_index}:{source_id}:{frame_index}",
                    "selector_index": selector_index,
                    "selector_kind": kind,
                    "source_id": source_id,
                    "video_path": sources[source_id]["video"],
                    "frame_index": frame_index,
                    "role": selector["role"],
                    "group": selector["group"],
                    "note": selector["note"],
                }
            )

    return {
        "task_path": task_path,
        "version": version,
        "sources": sources,
        "selectors": selectors,
        "items": items,
    }


def read_task_frame(item: dict[str, Any]):
    cap = cv2.VideoCapture(item["video_path"])
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {item['video_path']}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if item["frame_index"] >= total_frames:
        cap.release()
        raise RuntimeError(
            f"frame 越界: source={item['source_id']}, frame={item['frame_index']}, total_frames={total_frames}"
        )
    cap.set(cv2.CAP_PROP_POS_FRAMES, item["frame_index"])
    ret, frame = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError(
            f"无法读取指定帧: source={item['source_id']}, frame={item['frame_index']}"
        )
    return cap, frame
