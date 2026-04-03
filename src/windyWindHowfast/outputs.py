from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np

from .models import AnalysisResult, ROICandidate, ROIConfig, ROISelectionResult
from .roi_detection import save_roi_config
from .support import save_json


def draw_candidate_on_frame(
    frame: np.ndarray,
    candidate: ROICandidate,
    label: str,
    color: tuple[int, int, int],
) -> np.ndarray:
    canvas = frame.copy()
    center = (int(round(candidate.center_x)), int(round(candidate.center_y)))
    radius = max(1, int(round(candidate.radius)))
    cv2.circle(canvas, center, 7, color, -1)
    cv2.circle(canvas, center, radius, color, 3)
    cv2.putText(
        canvas,
        label,
        (max(5, center[0] - radius), max(20, center[1] - radius - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
        cv2.LINE_AA,
    )
    return canvas


def save_roi_debug_outputs(
    output_dir: Path,
    run_name: str,
    first_frame: np.ndarray,
    selection_result: ROISelectionResult,
    roi_debug: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates_path = output_dir / f"{run_name}_roi_candidates.json"
    detection_path = output_dir / f"{run_name}_roi_detection.json"
    best_debug_path = output_dir / f"{run_name}_roi_best_debug.png"

    save_json(candidates_path, selection_result.candidate_records)
    save_json(
        detection_path,
        {
            "status": selection_result.status,
            "source": selection_result.source,
            "fallback_used": selection_result.fallback_used,
            "failure_reason": selection_result.failure_reason,
            "reference_frame_strategy": selection_result.reference_frame_strategy,
            "reference_frame_indices": selection_result.reference_frame_indices,
            "selected_candidate": selection_result.selected_candidate,
            "selected_score": selection_result.selected_score,
            "candidate_count": len(selection_result.candidate_records),
            "metadata": selection_result.metadata,
        },
    )

    if selection_result.selected_candidate is not None:
        label = (
            f"{selection_result.source}:{selection_result.selected_score.total_score:.3f}"
            if selection_result.selected_score is not None
            else selection_result.source
        )
        best_canvas = draw_candidate_on_frame(
            frame=first_frame,
            candidate=selection_result.selected_candidate,
            label=label,
            color=(0, 255, 0),
        )
        cv2.imwrite(str(best_debug_path), best_canvas)

    if not roi_debug:
        return

    for record in selection_result.candidate_records[:10]:
        rank = record.get("rank")
        candidate_payload = record.get("candidate", {})
        if not isinstance(candidate_payload, dict):
            continue
        candidate = ROICandidate(
            center_x=float(candidate_payload.get("center_x", 0.0)),
            center_y=float(candidate_payload.get("center_y", 0.0)),
            radius=float(candidate_payload.get("radius", 1.0)),
            score=float(candidate_payload.get("score", 0.0)),
            source=str(candidate_payload.get("source", "unknown")),
            metadata=dict(candidate_payload.get("metadata", {})),
        )
        score_payload = record.get("score_breakdown", {})
        score = float(score_payload.get("total_score", 0.0)) if isinstance(score_payload, dict) else 0.0
        candidate_label = (
            f"#{rank} {candidate.source}:{score:.3f}"
            if rank is not None
            else f"{candidate.source}:{score:.3f}"
        )
        candidate_canvas = draw_candidate_on_frame(
            frame=first_frame,
            candidate=candidate,
            label=candidate_label,
            color=(0, 215, 255) if rank != 1 else (0, 255, 0),
        )
        candidate_path = output_dir / f"{run_name}_roi_candidate_{rank if rank is not None else 'manual'}.png"
        cv2.imwrite(str(candidate_path), candidate_canvas)


def save_analysis_outputs(
    output_dir: Path,
    run_name: str,
    first_frame: np.ndarray,
    roi_config: ROIConfig,
    figure: plt.Figure,
    result: AnalysisResult,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_with_roi = first_frame.copy()
    center = (roi_config.center_x, roi_config.center_y)
    cv2.circle(frame_with_roi, center, 8, (0, 0, 255), -1)
    cv2.circle(frame_with_roi, center, roi_config.radius, (0, 255, 0), 3)
    cv2.imwrite(str(output_dir / f"{run_name}_first_frame_with_roi.png"), frame_with_roi)

    figure.savefig(output_dir / f"{run_name}_analysis_summary.png", dpi=150, bbox_inches="tight")
    save_roi_config(roi_config, output_dir / f"{run_name}_roi.json")
    save_json(output_dir / f"{run_name}_analysis_result.json", result)
