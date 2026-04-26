from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from models import FEReferenceMode
from spectral import compute_mac, normalize_mode_shape

REQUIRED_FE_COLUMNS = {
    "basis",
    "mode_label",
    "frequency_hz",
    "damping_ratio",
    "point_1",
    "point_2",
    "point_3",
    "point_4",
    "point_5",
}


def load_fe_reference(path: Path | None) -> list[FEReferenceMode]:
    if path is None:
        return []
    frame = pd.read_csv(path)
    missing = sorted(REQUIRED_FE_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"FE 参考文件缺少必要列: {', '.join(missing)}")

    rows: list[FEReferenceMode] = []
    for _, row in frame.iterrows():
        shape = normalize_mode_shape(
            np.array([row[f"point_{index}"] for index in range(1, 6)], dtype=float)
        )
        rows.append(
            FEReferenceMode(
                basis=str(row["basis"]).strip(),
                mode_label=str(row["mode_label"]).strip(),
                frequency_hz=float(row["frequency_hz"]),
                damping_ratio=float(row["damping_ratio"]),
                shape=shape,
            )
        )
    return rows


def compare_modes_with_fe(
    *,
    case_id: int,
    identified_basis: str,
    identified_frequency_hz: float | None,
    identified_damping_ratio: float | None,
    identified_shape: np.ndarray | None,
    fe_modes: list[FEReferenceMode],
) -> list[dict[str, object]]:
    if not fe_modes or identified_shape is None or identified_shape.size == 0:
        return []

    comparisons: list[dict[str, object]] = []
    for mode in fe_modes:
        if mode.basis != identified_basis:
            continue
        comparisons.append(
            {
                "case_id": int(case_id),
                "basis": identified_basis,
                "mode_label": mode.mode_label,
                "identified_frequency_hz": identified_frequency_hz,
                "fe_frequency_hz": mode.frequency_hz,
                "frequency_error_hz": (
                    float(identified_frequency_hz - mode.frequency_hz)
                    if identified_frequency_hz is not None and np.isfinite(identified_frequency_hz)
                    else np.nan
                ),
                "identified_damping_ratio": identified_damping_ratio,
                "fe_damping_ratio": mode.damping_ratio,
                "damping_error": (
                    float(identified_damping_ratio - mode.damping_ratio)
                    if identified_damping_ratio is not None and np.isfinite(identified_damping_ratio)
                    else np.nan
                ),
                "mac": compute_mac(identified_shape, mode.shape),
            }
        )
    return comparisons
