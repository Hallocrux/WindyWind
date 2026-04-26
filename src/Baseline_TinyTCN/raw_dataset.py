from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.current.data_loading import DatasetRecord, QUALITY_COLUMNS, TIME_COLUMN
from src.current.features import WindowConfig


@dataclass(frozen=True)
class RawDataset:
    meta_df: pd.DataFrame
    windows: np.ndarray


def build_raw_window_dataset(
    records: list[DatasetRecord],
    cleaned_signal_frames: dict[int, pd.DataFrame],
    config: WindowConfig,
) -> RawDataset:
    meta_rows: list[dict[str, object]] = []
    window_arrays: list[np.ndarray] = []
    for record in records:
        signal_df = cleaned_signal_frames[record.case_id]
        numeric_columns = [
            column for column in signal_df.columns if column not in {TIME_COLUMN, *QUALITY_COLUMNS}
        ]
        for _, segment_df in signal_df.groupby("__segment_id", sort=True):
            total_rows = len(segment_df)
            if total_rows < config.window_size:
                continue
            segment_df = segment_df.reset_index(drop=True)
            for start in range(0, total_rows - config.window_size + 1, config.step_size):
                end = start + config.window_size
                window = segment_df.iloc[start:end]
                meta_rows.append(
                    {
                        "case_id": record.case_id,
                        "file_name": record.file_name,
                        "window_index": len(meta_rows),
                        "start_time": window[TIME_COLUMN].iloc[0],
                        "end_time": window[TIME_COLUMN].iloc[-1],
                        "wind_speed": record.wind_speed,
                        "rpm": record.rpm,
                    }
                )
                window_arrays.append(window[numeric_columns].to_numpy(dtype=float, copy=False).T.copy())
    if not window_arrays:
        raise ValueError("未构建出任何原始窗口样本。")
    return RawDataset(meta_df=pd.DataFrame(meta_rows), windows=np.stack(window_arrays, axis=0))
