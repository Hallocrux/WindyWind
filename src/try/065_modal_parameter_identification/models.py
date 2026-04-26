from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ModalConfig:
    sampling_rate: float = 50.0
    freq_min: float = 0.5
    freq_max: float = 6.0
    focus_min: float = 2.0
    focus_max: float = 3.0
    window_seconds: float = 20.0
    step_seconds: float = 10.0
    harmonic_orders: tuple[int, ...] = (1, 2, 3, 4)
    harmonic_half_width: float = 0.2
    ssi_block_rows: int = 30
    ssi_min_order: int = 2
    ssi_max_order: int = 20
    stabilization_freq_tol_hz: float = 0.15
    stabilization_damping_tol: float = 0.05
    stabilization_mac_tol: float = 0.8
    cluster_freq_tol_hz: float = 0.15
    max_damping_ratio: float = 0.2

    @property
    def window_size(self) -> int:
        return int(round(self.window_seconds * self.sampling_rate))

    @property
    def step_size(self) -> int:
        return int(round(self.step_seconds * self.sampling_rate))


@dataclass(frozen=True)
class ModalWindow:
    case_id: int
    window_index: int
    segment_id: int
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    data: np.ndarray


@dataclass(frozen=True)
class FEReferenceMode:
    basis: str
    mode_label: str
    frequency_hz: float
    damping_ratio: float
    shape: np.ndarray


@dataclass(frozen=True)
class CaseContext:
    case_id: int
    display_name: str
    rpm: float | None
    output_dir: Path
