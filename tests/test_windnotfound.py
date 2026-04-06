from __future__ import annotations

import unittest
import uuid
from pathlib import Path

import cv2
import numpy as np

from src.windNotFound.annotation_ui import AnnotationSession, AnnotationState, AnnotationStore
from src.windNotFound.fit_rpm import build_summary_payload, fit_selector_rpm
from src.windNotFound.task_io import load_task, resolve_jsonl_path
from src.windNotFound.video_rpm_eval import evaluate_task_video_rpm

TEST_RUNTIME_ROOT = Path("outputs/test_runtime").resolve()


def _make_test_video(path: Path, frame_count: int = 8, size: tuple[int, int] = (64, 48)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        size,
    )
    if not writer.isOpened():
        raise RuntimeError("测试视频写入失败。")
    for idx in range(frame_count):
        frame = np.full((size[1], size[0], 3), (idx * 20) % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def _make_runtime_dir(name: str) -> Path:
    TEST_RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
    path = TEST_RUNTIME_ROOT / f"{name}_{uuid.uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=True)
    return path


class WindNotFoundTaskTests(unittest.TestCase):
    def test_expand_selectors_keep_duplicate_frames(self) -> None:
        root = _make_runtime_dir("task_expand")
        video_path = root / "run01.avi"
        _make_test_video(video_path, frame_count=20)
        task_path = root / "task.yaml"
        task_path.write_text(
            "\n".join(
                [
                    "version: 1",
                    "sources:",
                    "  - id: run01",
                    "    video: run01.avi",
                    "selectors:",
                    "  - kind: window",
                    "    source: run01",
                    "    center: 10",
                    "    before: 1",
                    "    after: 1",
                    "    step: 1",
                    "    role: anchor",
                    "    group: seg_01",
                    "  - kind: explicit",
                    "    source: run01",
                    "    frames: [20, 9]",
                    "    role: bridge",
                ]
            ),
            encoding="utf-8",
        )
        task = load_task(task_path)
        self.assertEqual([item["frame_index"] for item in task["items"]], [9, 10, 11, 9, 20])
        self.assertEqual(task["items"][0]["task_item_id"], "0:run01:9")
        self.assertEqual(task["items"][3]["task_item_id"], "1:run01:9")

    def test_invalid_frame_raises_when_session_loads(self) -> None:
        root = _make_runtime_dir("invalid_frame")
        video_path = root / "run01.avi"
        _make_test_video(video_path, frame_count=3)
        task_path = root / "task.yaml"
        task_path.write_text(
            "\n".join(
                [
                    "version: 1",
                    "sources:",
                    "  - id: run01",
                    "    video: run01.avi",
                    "selectors:",
                    "  - kind: explicit",
                    "    source: run01",
                    "    frames: [5]",
                ]
            ),
            encoding="utf-8",
        )
        session = AnnotationSession(task_path=task_path)
        with self.assertRaises(RuntimeError):
            session._load_item(0)

    def test_session_uses_auto_output_paths(self) -> None:
        root = _make_runtime_dir("auto_paths")
        video_path = root / "run01.avi"
        _make_test_video(video_path, frame_count=5)
        task_path = root / "my_task.yaml"
        task_path.write_text(
            "\n".join(
                [
                    "version: 1",
                    "sources:",
                    "  - id: run01",
                    "    video: run01.avi",
                    "selectors:",
                    "  - kind: explicit",
                    "    source: run01",
                    "    frames: [1]",
                ]
            ),
            encoding="utf-8",
        )
        session = AnnotationSession(task_path=task_path, readonly=True)
        self.assertEqual(session.jsonl_path, resolve_jsonl_path(task_path))
        self.assertTrue(session.summary_path.exists())


class WindNotFoundStoreAndFitTests(unittest.TestCase):
    def test_store_append_and_latest_record(self) -> None:
        root = _make_runtime_dir("store_append")
        video_path = root / "run01.avi"
        _make_test_video(video_path, frame_count=5)
        task_path = root / "task.yaml"
        task_path.write_text(
            "\n".join(
                [
                    "version: 1",
                    "sources:",
                    "  - id: run01",
                    "    video: run01.avi",
                    "selectors:",
                    "  - kind: explicit",
                    "    source: run01",
                    "    frames: [1]",
                ]
            ),
            encoding="utf-8",
        )
        task = load_task(task_path)
        store = AnnotationStore(root / "annotations.jsonl")
        first_points = {
            "support_a": {"x": 1, "y": 1},
            "support_b": {"x": 2, "y": 2},
            "center": {"x": 3, "y": 3},
            "blade_1": {"x": 4, "y": 4},
            "blade_2": {"x": 5, "y": 5},
            "blade_3": {"x": 6, "y": 6},
        }
        second_points = dict(first_points)
        second_points["blade_3"] = {"x": 10, "y": 10}
        store.append(task, task["items"][0], first_points)
        store.append(task, task["items"][0], second_points)

        reloaded = AnnotationStore(root / "annotations.jsonl")
        self.assertEqual(reloaded.get(task["items"][0]["task_item_id"])["points"]["blade_3"], {"x": 10, "y": 10})

    def test_invalid_jsonl_line_is_reported(self) -> None:
        root = _make_runtime_dir("bad_jsonl")
        path = root / "annotations.jsonl"
        path.write_text("{bad json}\n", encoding="utf-8")
        store = AnnotationStore(path)
        self.assertEqual(len(store.invalid_lines), 1)

    def test_fit_selector_rpm_uses_annotation_points_directly(self) -> None:
        fps = 10.0
        center = {"x": 2000, "y": 2000}
        radius = 1000
        records = []
        for frame_index, angle_deg in enumerate([0.0, 36.0, 72.0]):
            angle_rad = np.deg2rad(angle_deg)
            blade_1 = {
                "x": int(round(center["x"] + radius * np.cos(angle_rad))),
                "y": int(round(center["y"] + radius * np.sin(angle_rad))),
            }
            records.append(
                {
                    "frame_index": frame_index,
                    "points": {
                        "center": center,
                        "blade_1": blade_1,
                    },
                }
            )
        result = fit_selector_rpm(records, fps)
        self.assertEqual(result["status"], "ok")
        self.assertAlmostEqual(result["rpm"], 60.0, places=1)

    def test_summary_payload_contains_selector_rpm(self) -> None:
        root = _make_runtime_dir("summary_fit_rpm")
        video_path = root / "run01.avi"
        _make_test_video(video_path, frame_count=8)
        task_path = root / "task.yaml"
        task_path.write_text(
            "\n".join(
                [
                    "version: 1",
                    "sources:",
                    "  - id: run01",
                    "    video: run01.avi",
                    "selectors:",
                    "  - kind: explicit",
                    "    source: run01",
                    "    frames: [0, 1, 2]",
                ]
            ),
            encoding="utf-8",
        )
        task = load_task(task_path)
        store = AnnotationStore(root / "annotations.jsonl")
        center = {"x": 2000, "y": 2000}
        radius = 1000
        for frame_index, angle_deg in enumerate([0.0, 36.0, 72.0]):
            angle_rad = np.deg2rad(angle_deg)
            blade_1 = {
                "x": int(round(center["x"] + radius * np.cos(angle_rad))),
                "y": int(round(center["y"] + radius * np.sin(angle_rad))),
            }
            points = {
                "support_a": {"x": 0, "y": 0},
                "support_b": {"x": 1, "y": 1},
                "center": center,
                "blade_1": blade_1,
                "blade_2": {"x": 10, "y": 10},
                "blade_3": {"x": 20, "y": 20},
            }
            store.append(task, task["items"][frame_index], points)

        payload = build_summary_payload(task, store.by_task_item_id, root / "annotations.jsonl")
        selector_summary = payload["selectors"][0]
        self.assertEqual(payload["summary_kind"], "annotation_fit_rpm")
        self.assertEqual(selector_summary["status"], "ok")
        self.assertAlmostEqual(selector_summary["rpm"], 60.0, places=1)


class WindNotFoundStateTests(unittest.TestCase):
    def test_undo_chain_and_reload(self) -> None:
        state = AnnotationState()
        for idx in range(6):
            state.add_point(idx, idx)
        self.assertTrue(state.is_complete)
        self.assertTrue(state.undo())
        self.assertFalse(state.is_complete)
        self.assertEqual(state.current_point_name, "blade_3")
        reloaded = {
            "support_a": {"x": 1, "y": 1},
            "support_b": {"x": 2, "y": 2},
            "center": {"x": 3, "y": 3},
            "blade_1": {"x": 4, "y": 4},
            "blade_2": {"x": 5, "y": 5},
            "blade_3": {"x": 6, "y": 6},
        }
        state.load_existing(reloaded)
        self.assertTrue(state.history_loaded)
        self.assertTrue(state.is_complete)
        self.assertTrue(state.undo())
        self.assertEqual(state.current_point_name, "blade_3")


class WindNotFoundVideoEvalTests(unittest.TestCase):
    def test_video_eval_uses_per_frame_roi(self) -> None:
        root = _make_runtime_dir("video_eval")
        video_path = root / "rotor.avi"
        frame_size = (128, 128)
        center = (64, 64)
        radius = 30
        angles = [idx * 30.0 for idx in range(12)]
        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"MJPG"),
            10.0,
            frame_size,
        )
        if not writer.isOpened():
            raise RuntimeError("测试视频写入失败。")
        for angle_deg in angles:
            frame = np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8)
            angle_rad = np.deg2rad(angle_deg)
            end_point = (
                int(round(center[0] + radius * np.cos(angle_rad))),
                int(round(center[1] + radius * np.sin(angle_rad))),
            )
            cv2.line(frame, center, end_point, (255, 255, 255), 3)
            writer.write(frame)
        writer.release()

        task_path = root / "task.yaml"
        task_path.write_text(
            "\n".join(
                [
                    "version: 1",
                    "sources:",
                    "  - id: run01",
                    "    video: rotor.avi",
                    "selectors:",
                    "  - kind: explicit",
                    "    source: run01",
                    "    frames: [0,1,2,3,4,5,6,7,8,9,10,11]",
                ]
            ),
            encoding="utf-8",
        )
        task = load_task(task_path)
        store = AnnotationStore(resolve_jsonl_path(task_path))
        for frame_index, angle_deg in enumerate(angles):
            angle_rad = np.deg2rad(angle_deg)
            blade_1 = {
                "x": int(round(center[0] + radius * np.cos(angle_rad))),
                "y": int(round(center[1] + radius * np.sin(angle_rad))),
            }
            points = {
                "support_a": {"x": 0, "y": 0},
                "support_b": {"x": 1, "y": 1},
                "center": {"x": center[0], "y": center[1]},
                "blade_1": blade_1,
                "blade_2": {"x": 0, "y": 0},
                "blade_3": {"x": 0, "y": 0},
            }
            store.append(task, task["items"][frame_index], points)

        result = evaluate_task_video_rpm(task, selector_index=0)
        payload = result["payload"]["result"]
        self.assertEqual(payload["status"], "ok")
        self.assertAlmostEqual(payload["manual_fit"]["rpm"], 50.0, places=1)
        self.assertAlmostEqual(payload["video_fft"]["rpm"], 50.0, places=1)


if __name__ == "__main__":
    unittest.main()
