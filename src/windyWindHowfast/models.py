from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class ROIConfig:
    center_x: int
    center_y: int
    radius: int


@dataclass
class ROICandidate:
    center_x: float
    center_y: float
    radius: float
    score: float
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ROICandidateScore:
    total_score: float
    center_consistency_score: float
    radius_consistency_score: float
    signal_energy_score: float
    boundary_penalty: float
    occlusion_penalty: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ROIReferenceFrames:
    strategy: str
    frame_indices: list[int]
    frames: list[np.ndarray]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ROIReferenceContext:
    frames: list[np.ndarray]
    frame_indices: list[int]
    gray_frames: list[np.ndarray]
    edge_frames: list[np.ndarray]
    motion_maps: list[np.ndarray]
    combined_frames: list[np.ndarray]
    mean_gray: np.ndarray
    mean_edges: np.ndarray
    mean_motion: np.ndarray
    combined_map: np.ndarray
    height: int
    width: int


@dataclass
class ROISelectionResult:
    status: str
    source: str
    selected_candidate: ROICandidate | None
    selected_score: ROICandidateScore | None
    candidate_records: list[dict[str, Any]]
    failure_reason: str | None = None
    fallback_used: bool = False
    reference_frame_strategy: str | None = None
    reference_frame_indices: list[int] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalysisResult:
    video_path: str
    fps: float
    total_frames: int
    start_frame: int
    analyzed_frames: int
    analyzed_duration_sec: float
    angular_res: int
    inner_radius_ratio: float
    min_temporal_hz: float
    max_temporal_hz: float
    max_spatial_mode: int
    roi: dict[str, int]
    roi_source: str
    roi_detection_status: str
    roi_score: float | None
    roi_reference_frame_strategy: str | None
    roi_reference_frame_indices: list[int]
    peak_temporal_hz: float
    peak_spatial_mode: int
    peak_magnitude: float
    rotor_freq_hz: float
    rpm: float
