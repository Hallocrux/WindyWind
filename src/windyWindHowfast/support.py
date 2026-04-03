from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def normalize_map(data: np.ndarray) -> np.ndarray:
    arr = data.astype(np.float32)
    min_val = float(np.min(arr))
    max_val = float(np.max(arr))
    if max_val - min_val < 1e-6:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - min_val) / (max_val - min_val)


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "__dataclass_fields__"):
        return jsonable(asdict(value))
    return value


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def compute_edge_map(gray: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    return edges.astype(np.float32) / 255.0


def downscale_gray(gray: np.ndarray, max_side: int = 320) -> np.ndarray:
    h, w = gray.shape[:2]
    scale = max(h, w) / max_side
    if scale <= 1.0:
        return gray
    new_w = max(1, int(round(w / scale)))
    new_h = max(1, int(round(h / scale)))
    return cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)


def read_frames_at_indices(cap: cv2.VideoCapture, frame_indices: list[int]) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    last_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    for frame_index in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.set(cv2.CAP_PROP_POS_FRAMES, last_pos)
    return frames
