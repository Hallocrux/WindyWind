from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data_loading import DatasetRecord, QUALITY_COLUMNS, TIME_COLUMN

WINDOW_META_COLUMNS = [
    "case_id",
    "file_name",
    "window_index",
    "start_time",
    "end_time",
    "wind_speed",
    "rpm",
    "raw_missing_ratio",
    "raw_missing_rows",
    "touches_leading_missing",
    "touches_trailing_missing",
]
BANDS = ((0.0, 2.0), (2.0, 5.0), (5.0, 10.0))


@dataclass(frozen=True)
class WindowConfig:
    sampling_rate: float = 50.0
    window_size: int = 250
    step_size: int = 125


def build_case_feature_frame(
    record: DatasetRecord,
    signal_df: pd.DataFrame,
    config: WindowConfig,
) -> pd.DataFrame:
    numeric_columns = [
        column
        for column in signal_df.columns
        if column not in {TIME_COLUMN, *QUALITY_COLUMNS}
    ]
    total_rows = len(signal_df)
    rows: list[dict[str, float | int | str | pd.Timestamp | None]] = []

    for start in range(0, total_rows - config.window_size + 1, config.step_size):
        end = start + config.window_size
        window = signal_df.iloc[start:end]

        feature_row: dict[str, float | int | str | pd.Timestamp | None] = {
            "case_id": record.case_id,
            "file_name": record.file_name,
            "window_index": len(rows),
            "start_time": window[TIME_COLUMN].iloc[0],
            "end_time": window[TIME_COLUMN].iloc[-1],
            "wind_speed": record.wind_speed,
            "rpm": record.rpm,
            "raw_missing_ratio": float(
                window["__row_missing_count"].sum()
                / (config.window_size * len(numeric_columns))
            ),
            "raw_missing_rows": int(window["__row_has_missing"].sum()),
            "touches_leading_missing": int(window["__in_leading_missing_block"].any()),
            "touches_trailing_missing": int(window["__in_trailing_missing_block"].any()),
        }

        for column in numeric_columns:
            feature_row.update(
                _extract_channel_features(
                    window[column].to_numpy(dtype=float, copy=False),
                    column,
                    config.sampling_rate,
                )
            )

        rows.append(feature_row)

    if not rows:
        raise ValueError(
            f"{record.file_name} 行数不足以切出窗口，当前行数={total_rows}。"
        )

    return pd.DataFrame(rows)


def get_vibration_feature_columns(feature_df: pd.DataFrame) -> list[str]:
    return [column for column in feature_df.columns if column not in WINDOW_META_COLUMNS]


def _extract_channel_features(
    signal: np.ndarray,
    prefix: str,
    sampling_rate: float,
) -> dict[str, float]:
    centered = signal - float(np.mean(signal))
    rms = float(np.sqrt(np.mean(np.square(signal))))
    feature_map: dict[str, float] = {
        f"{prefix}__mean": float(np.mean(signal)),
        f"{prefix}__std": float(np.std(signal, ddof=0)),
        f"{prefix}__min": float(np.min(signal)),
        f"{prefix}__max": float(np.max(signal)),
        f"{prefix}__ptp": float(np.ptp(signal)),
        f"{prefix}__rms": rms,
    }

    spectrum = np.fft.rfft(centered)
    magnitudes = np.abs(spectrum)
    power = np.square(magnitudes)
    freqs = np.fft.rfftfreq(signal.size, d=1.0 / sampling_rate)

    if len(magnitudes) > 1:
        peak_index = int(np.argmax(magnitudes[1:]) + 1)
        peak_frequency = float(freqs[peak_index])
        peak_amplitude = float(magnitudes[peak_index])
        total_energy = float(np.sum(power[1:]))
    else:
        peak_frequency = 0.0
        peak_amplitude = 0.0
        total_energy = 0.0

    feature_map[f"{prefix}__fft_peak_freq"] = peak_frequency
    feature_map[f"{prefix}__fft_peak_amp"] = peak_amplitude
    feature_map[f"{prefix}__fft_total_energy"] = total_energy

    for low, high in BANDS:
        mask = (freqs >= low) & (freqs < high)
        band_energy = float(np.sum(power[mask]))
        ratio = band_energy / total_energy if total_energy > 0 else 0.0
        band_name = f"{int(low)}_{int(high)}hz"
        feature_map[f"{prefix}__fft_band_ratio_{band_name}"] = ratio

    return feature_map
