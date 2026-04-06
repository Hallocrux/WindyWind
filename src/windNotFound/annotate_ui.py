from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

try:
    from task_io import (
        ANNOTATION_POINT_ORDER,
        POINT_INSTRUCTIONS,
        load_task,
        read_task_frame,
        resolve_jsonl_path,
        resolve_output_dir,
        resolve_summary_path,
    )
    from fit_rpm import build_summary_payload
except ModuleNotFoundError:
    from src.windNotFound.task_io import (
        ANNOTATION_POINT_ORDER,
        POINT_INSTRUCTIONS,
        load_task,
        read_task_frame,
        resolve_jsonl_path,
        resolve_output_dir,
        resolve_summary_path,
    )
    from src.windNotFound.fit_rpm import build_summary_payload


POINT_COLORS = {
    "support_a": (0, 165, 255),
    "support_b": (0, 215, 255),
    "center": (0, 0, 255),
    "blade_1": (0, 255, 0),
    "blade_2": (255, 0, 0),
    "blade_3": (255, 255, 0),
}

WINDOW_NAME = "Windmill Annotation"


class AnnotationStore:
    def __init__(self, jsonl_path: str | Path) -> None:
        self.jsonl_path = Path(jsonl_path).resolve()
        self.by_task_item_id: dict[str, dict] = {}
        self.invalid_lines: list[tuple[int, str]] = []
        self._load()

    def _load(self) -> None:
        if not self.jsonl_path.exists():
            return
        with self.jsonl_path.open("r", encoding="utf-8") as fp:
            for line_number, raw_line in enumerate(fp, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                    for name in ANNOTATION_POINT_ORDER:
                        point = payload["points"][name]
                        if not isinstance(point["x"], int) or not isinstance(point["y"], int):
                            raise ValueError(f"points.{name} 必须包含整数 x/y。")
                except Exception as exc:
                    self.invalid_lines.append((line_number, str(exc)))
                    continue
                self.by_task_item_id[str(payload["task_item_id"])] = payload

    def get(self, task_item_id: str) -> dict | None:
        return self.by_task_item_id.get(task_item_id)

    def append(self, task: dict, item: dict, points: dict[str, dict[str, int]]) -> dict:
        payload = {
            "task_yaml": str(task["task_path"]),
            "task_version": task["version"],
            "task_item_id": item["task_item_id"],
            "selector_index": item["selector_index"],
            "selector_kind": item["selector_kind"],
            "source_id": item["source_id"],
            "video_path": item["video_path"],
            "frame_index": item["frame_index"],
            "role": item["role"],
            "group": item["group"],
            "note": item["note"],
            "annotation_schema_version": 1,
            "annotated_at": datetime.now().isoformat(timespec="seconds"),
            "points": points,
        }
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with self.jsonl_path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(payload, ensure_ascii=False) + "\n")
            fp.flush()
        self.by_task_item_id[item["task_item_id"]] = payload
        return payload


class AnnotationState:
    def __init__(self) -> None:
        self.points: dict[str, dict[str, int]] = {}
        self.history_loaded = False

    @property
    def current_point_name(self) -> str | None:
        if self.is_complete:
            return None
        return ANNOTATION_POINT_ORDER[len(self.points)]

    @property
    def is_complete(self) -> bool:
        return len(self.points) == len(ANNOTATION_POINT_ORDER)

    def load_existing(self, points: dict[str, dict[str, int]]) -> None:
        self.points = {name: {"x": int(p["x"]), "y": int(p["y"])} for name, p in points.items()}
        self.history_loaded = True

    def add_point(self, x: int, y: int) -> None:
        if self.is_complete:
            raise ValueError("当前帧标注已完成，不能继续加点。")
        self.points[self.current_point_name or ""] = {"x": x, "y": y}

    def undo(self) -> bool:
        if not self.points:
            return False
        last_key = ANNOTATION_POINT_ORDER[len(self.points) - 1]
        self.points.pop(last_key, None)
        return True


class AnnotationSession:
    def __init__(
        self,
        task_path: str | Path,
        *,
        start_at: str | None = None,
        readonly: bool = False,
        show_done: bool = False,
    ) -> None:
        self.task = load_task(task_path)
        self.output_dir = resolve_output_dir(self.task["task_path"])
        self.jsonl_path = resolve_jsonl_path(self.task["task_path"])
        self.summary_path = resolve_summary_path(self.task["task_path"])
        self.store = AnnotationStore(self.jsonl_path)
        if self.store.invalid_lines:
            bad_lines = ", ".join(str(line_no) for line_no, _ in self.store.invalid_lines[:5])
            raise ValueError(f"jsonl contains invalid lines and cannot be used. Example line numbers: {bad_lines}")
        self._write_summary()
        self.readonly = readonly
        self.show_done = show_done
        self.current_index = self._resolve_start_index(start_at)
        self.state = AnnotationState()
        self.current_item: dict | None = None
        self.frame: np.ndarray | None = None
        self.scale = 1.0
        self.cap = None
        self.exit_requested = False

    def _resolve_start_index(self, start_at: str | None) -> int:
        if not self.task["items"]:
            raise ValueError("The task is empty. No frames are available for annotation.")
        if start_at is None:
            if self.show_done:
                return 0
            for idx, item in enumerate(self.task["items"]):
                if self.store.get(item["task_item_id"]) is None:
                    return idx
            return len(self.task["items"]) - 1
        if start_at.isdigit():
            index = int(start_at)
            if not 0 <= index < len(self.task["items"]):
                raise ValueError(f"start-at index is out of range: {index}")
            return index
        for idx, item in enumerate(self.task["items"]):
            if item["task_item_id"] == start_at:
                return idx
        raise ValueError(f"Cannot find the task_item_id specified by start-at: {start_at}")

    def _load_item(self, index: int) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        item = self.task["items"][index]
        cap, frame = read_task_frame(item)
        self.cap = cap
        self.current_item = item
        self.frame = frame
        self.state = AnnotationState()
        stored = self.store.get(item["task_item_id"])
        if stored is not None:
            self.state.load_existing(stored["points"])

    def on_click(self, event: int, x: int, y: int, flags: int, param: object) -> None:
        _ = flags, param
        if event != cv2.EVENT_LBUTTONDOWN or self.readonly or self.frame is None or self.scale <= 0:
            return
        self.state.add_point(int(round(x / self.scale)), int(round(y / self.scale)))
        if self.state.is_complete:
            self._persist_current_item()
            self._advance_after_save()

    def _persist_current_item(self) -> None:
        if self.current_item is None:
            raise RuntimeError("No active task item is loaded.")
        self.store.append(self.task, self.current_item, self.state.points)
        self._write_summary()

    def _write_summary(self) -> None:
        payload = build_summary_payload(
            task=self.task,
            annotations_by_task_item_id=self.store.by_task_item_id,
            jsonl_path=self.jsonl_path,
        )
        self.summary_path.parent.mkdir(parents=True, exist_ok=True)
        self.summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _advance_after_save(self) -> None:
        if self.current_index >= len(self.task["items"]) - 1:
            self.exit_requested = True
            return
        self.current_index += 1
        self._load_item(self.current_index)

    def _build_info_lines(self) -> list[str]:
        assert self.current_item is not None
        status = (
            "Existing annotation is loaded. Press z to redo from the last point."
            if self.state.history_loaded and self.state.is_complete and not self.readonly
            else POINT_INSTRUCTIONS.get(self.state.current_point_name or "", "Current frame is complete.")
        )
        return [
            f"Progress: frame {self.current_index + 1} / {len(self.task['items'])}",
            f"source={self.current_item['source_id']} frame={self.current_item['frame_index']} selector={self.current_item['selector_index']}:{self.current_item['selector_kind']}",
            f"group={self.current_item['group'] or '-'} role={self.current_item['role'] or '-'} note={self.current_item['note'] or '-'}",
            status,
            "Keys: left click add point | z undo/redo | n next if current item already exists | p previous view | q/Esc quit",
            "Rule: blade_1 is the marker blade; blade_2 and blade_3 must increase clockwise.",
        ]

    def _draw_overlay(self) -> np.ndarray:
        assert self.frame is not None
        canvas = self.frame.copy()
        for idx, line in enumerate(self._build_info_lines()):
            y = 28 + idx * 24
            cv2.putText(canvas, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(canvas, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30, 30, 30), 1, cv2.LINE_AA)
        for name in ANNOTATION_POINT_ORDER:
            point = self.state.points.get(name)
            if point is None:
                continue
            color = POINT_COLORS[name]
            cv2.circle(canvas, (point["x"], point["y"]), 6, color, -1)
            cv2.putText(canvas, name, (point["x"] + 8, point["y"] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        return canvas

    def _show_frame(self) -> None:
        canvas = self._draw_overlay()
        frame_height = canvas.shape[0]
        self.scale = 900 / frame_height if frame_height > 900 else 1.0
        display = cv2.resize(canvas, None, fx=self.scale, fy=self.scale) if self.scale != 1.0 else canvas
        cv2.imshow(WINDOW_NAME, display)

    def _move(self, new_index: int) -> None:
        if not 0 <= new_index < len(self.task["items"]):
            return
        self.current_index = new_index
        self._load_item(self.current_index)

    def run(self) -> None:
        cv2.namedWindow(WINDOW_NAME)
        cv2.setMouseCallback(WINDOW_NAME, self.on_click)
        self._load_item(self.current_index)
        while True:
            self._show_frame()
            key = cv2.waitKey(20) & 0xFF
            if key == 255:
                if self.exit_requested:
                    break
                continue
            if key in {27, ord("q")}:
                break
            if key == ord("z") and not self.readonly:
                self.state.undo()
                continue
            if key == ord("n"):
                if self.current_item is not None and self.store.get(self.current_item["task_item_id"]) is not None:
                    self._move(self.current_index + 1)
                continue
            if key == ord("p"):
                self._move(self.current_index - 1)
                continue
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        cv2.destroyAllWindows()
