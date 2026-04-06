from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.current.data_loading import DatasetRecord, QUALITY_COLUMNS, TIME_COLUMN
from src.current.features import WindowConfig

TRY_NAME = "009_phase1_feature_groups"
REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_ROOT = REPO_ROOT / "outputs" / "try" / TRY_NAME

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
QUALITY_FEATURE_COLUMNS = [
    "raw_missing_ratio",
    "raw_missing_rows",
    "touches_leading_missing",
    "touches_trailing_missing",
]
BASELINE_SUFFIXES = ["raw_mean", "raw_median"]
BASE_DYNAMIC_SUFFIXES = [
    "dyn_std",
    "dyn_min",
    "dyn_max",
    "dyn_ptp",
    "dyn_rms",
    "dyn_fft_peak_freq",
    "dyn_fft_peak_amp",
    "dyn_fft_total_energy",
    "dyn_fft_band_ratio_0_2hz",
    "dyn_fft_band_ratio_2_5hz",
    "dyn_fft_band_ratio_5_10hz",
]
ROBUST_SUFFIXES = [
    "dyn_iqr",
    "dyn_mad",
    "dyn_skewness",
    "dyn_excess_kurtosis",
    "dyn_zero_crossing_rate",
    "dyn_crest_factor",
]
FREQ_SHAPE_SUFFIXES = [
    "dyn_fft_top1_freq",
    "dyn_fft_top2_freq",
    "dyn_fft_top3_freq",
    "dyn_fft_top1_amp",
    "dyn_fft_top2_amp",
    "dyn_fft_top3_amp",
    "dyn_spectral_centroid",
    "dyn_spectral_bandwidth",
    "dyn_spectral_entropy",
    "dyn_spectral_rolloff_85",
    "dyn_fft_band_ratio_0_1hz",
    "dyn_fft_band_ratio_1_2hz",
    "dyn_fft_band_ratio_2_3hz",
    "dyn_fft_band_ratio_3_5hz",
    "dyn_fft_band_ratio_5_8hz",
    "dyn_fft_band_ratio_8_10hz",
]
CROSS_CHANNEL_COLUMNS = [
    "acc_rms_mean",
    "acc_rms_std",
    "acc_rms_range",
    "strain_rms_mean",
    "strain_rms_std",
    "strain_rms_range",
    "acc_peak_freq_mean",
    "acc_peak_freq_std",
    "acc_peak_freq_range",
    "strain_peak_freq_mean",
    "strain_peak_freq_std",
    "strain_peak_freq_range",
    "acc_energy_sum",
    "acc_energy_mean",
    "acc_energy_std",
    "strain_energy_sum",
    "strain_energy_mean",
    "strain_energy_std",
    "acc_corr_mean",
    "acc_corr_std",
    "acc_corr_max_abs",
    "strain_corr_mean",
    "strain_corr_std",
    "strain_corr_max_abs",
    "acc_energy_to_strain_energy",
    "acc_rms_mean_to_strain_rms_mean",
]
PRIMARY_FREQ_BANDS = ((0.0, 2.0), (2.0, 5.0), (5.0, 10.0))
SECONDARY_FREQ_BANDS = ((0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 5.0), (5.0, 8.0), (8.0, 10.0))


@dataclass(frozen=True)
class GroupSpec:
    group_name: str
    include_robust_time: bool = False
    include_freq_shape: bool = False
    include_cross_channel: bool = False
    include_quality: bool = False


@dataclass(frozen=True)
class PromotionDecision:
    promoted: bool
    fail_reasons: list[str]


@dataclass(frozen=True)
class Phase1RuntimeConfig:
    max_workers: int = 1
    rf_n_jobs: int = 1


GROUP_SPECS = [
    GroupSpec("G0_BASE"),
    GroupSpec("G1_ROBUST_TIME", include_robust_time=True),
    GroupSpec("G2_FREQ_SHAPE", include_freq_shape=True),
    GroupSpec("G3_CROSS_CHANNEL", include_cross_channel=True),
    GroupSpec("G4_QUALITY", include_quality=True),
    GroupSpec("G5_TIME_PLUS_FREQ", include_robust_time=True, include_freq_shape=True),
    GroupSpec(
        "G6_TIME_FREQ_CROSS",
        include_robust_time=True,
        include_freq_shape=True,
        include_cross_channel=True,
    ),
    GroupSpec(
        "G7_ALL",
        include_robust_time=True,
        include_freq_shape=True,
        include_cross_channel=True,
        include_quality=True,
    ),
]
GROUP_SPEC_MAP = {spec.group_name: spec for spec in GROUP_SPECS}


def build_feature_frame(
    records: list[DatasetRecord],
    cleaned_signal_frames: dict[int, pd.DataFrame],
    config: WindowConfig,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str | pd.Timestamp | None]] = []
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

                channel_caches: dict[str, dict[str, object]] = {}
                for column in numeric_columns:
                    raw_signal = window[column].to_numpy(dtype=float, copy=False)
                    channel_features, channel_cache = extract_channel_feature_superset(
                        raw_signal,
                        prefix=column,
                        sampling_rate=config.sampling_rate,
                    )
                    feature_row.update(channel_features)
                    channel_caches[column] = channel_cache

                feature_row.update(build_cross_channel_features(channel_caches))
                rows.append(feature_row)

    if not rows:
        raise ValueError("第一阶段探索未生成任何窗口。")
    return pd.DataFrame(rows)


def extract_channel_feature_superset(
    raw_signal: np.ndarray,
    prefix: str,
    sampling_rate: float,
) -> tuple[dict[str, float], dict[str, object]]:
    raw_mean = float(np.mean(raw_signal))
    raw_median = float(np.median(raw_signal))
    dynamic_signal = raw_signal - raw_mean

    feature_map: dict[str, float] = {
        f"{prefix}__raw_mean": raw_mean,
        f"{prefix}__raw_median": raw_median,
    }

    feature_map.update(extract_dynamic_base_features(dynamic_signal, prefix, sampling_rate))
    feature_map.update(extract_dynamic_robust_features(dynamic_signal, prefix))
    feature_map.update(extract_dynamic_frequency_shape_features(dynamic_signal, prefix, sampling_rate))

    cache = {
        "group": get_channel_group(prefix),
        "dynamic_signal": dynamic_signal,
        "rms": feature_map[f"{prefix}__dyn_rms"],
        "peak_freq": feature_map[f"{prefix}__dyn_fft_peak_freq"],
        "energy": feature_map[f"{prefix}__dyn_fft_total_energy"],
    }
    return feature_map, cache


def extract_dynamic_base_features(
    dynamic_signal: np.ndarray,
    prefix: str,
    sampling_rate: float,
) -> dict[str, float]:
    rms = float(np.sqrt(np.mean(np.square(dynamic_signal))))
    feature_map: dict[str, float] = {
        f"{prefix}__dyn_std": float(np.std(dynamic_signal, ddof=0)),
        f"{prefix}__dyn_min": float(np.min(dynamic_signal)),
        f"{prefix}__dyn_max": float(np.max(dynamic_signal)),
        f"{prefix}__dyn_ptp": float(np.ptp(dynamic_signal)),
        f"{prefix}__dyn_rms": rms,
    }

    spectrum_info = _compute_spectrum_info(dynamic_signal, sampling_rate)
    feature_map[f"{prefix}__dyn_fft_peak_freq"] = spectrum_info["peak_freq"]
    feature_map[f"{prefix}__dyn_fft_peak_amp"] = spectrum_info["peak_amp"]
    feature_map[f"{prefix}__dyn_fft_total_energy"] = spectrum_info["total_energy"]
    feature_map.update(
        _band_ratio_features(
            prefix=prefix,
            spectrum_info=spectrum_info,
            bands=PRIMARY_FREQ_BANDS,
        )
    )
    return feature_map


def extract_dynamic_robust_features(dynamic_signal: np.ndarray, prefix: str) -> dict[str, float]:
    q25 = float(np.quantile(dynamic_signal, 0.25))
    q75 = float(np.quantile(dynamic_signal, 0.75))
    median = float(np.median(dynamic_signal))
    std = float(np.std(dynamic_signal, ddof=0))
    rms = float(np.sqrt(np.mean(np.square(dynamic_signal))))
    centered = dynamic_signal / std if std > 0 else np.zeros_like(dynamic_signal)
    return {
        f"{prefix}__dyn_iqr": q75 - q25,
        f"{prefix}__dyn_mad": float(np.median(np.abs(dynamic_signal - median))),
        f"{prefix}__dyn_skewness": float(np.mean(np.power(centered, 3))) if std > 0 else 0.0,
        f"{prefix}__dyn_excess_kurtosis": float(np.mean(np.power(centered, 4)) - 3.0) if std > 0 else 0.0,
        f"{prefix}__dyn_zero_crossing_rate": zero_crossing_rate(dynamic_signal),
        f"{prefix}__dyn_crest_factor": float(np.max(np.abs(dynamic_signal)) / rms) if rms > 0 else 0.0,
    }


def extract_dynamic_frequency_shape_features(
    dynamic_signal: np.ndarray,
    prefix: str,
    sampling_rate: float,
) -> dict[str, float]:
    spectrum_info = _compute_spectrum_info(dynamic_signal, sampling_rate)
    freqs = spectrum_info["freqs"]
    power = spectrum_info["power"]
    total_energy = spectrum_info["total_energy"]
    top_freqs = spectrum_info["top_freqs"]
    top_amps = spectrum_info["top_amps"]
    if total_energy <= 0.0:
        centroid = 0.0
        bandwidth = 0.0
        entropy = 0.0
        rolloff = 0.0
    else:
        centroid = float(np.sum(freqs * power) / total_energy)
        bandwidth = float(np.sqrt(np.sum(np.square(freqs - centroid) * power) / total_energy))
        probability = power / total_energy
        non_zero_probability = probability[probability > 0]
        entropy = float(-np.sum(non_zero_probability * np.log(non_zero_probability))) if len(non_zero_probability) else 0.0
        cumulative = np.cumsum(power)
        rolloff_idx = int(np.searchsorted(cumulative, 0.85 * total_energy, side="left"))
        rolloff_idx = min(rolloff_idx, len(freqs) - 1)
        rolloff = float(freqs[rolloff_idx]) if len(freqs) else 0.0

    feature_map = {
        f"{prefix}__dyn_fft_top1_freq": top_freqs[0],
        f"{prefix}__dyn_fft_top2_freq": top_freqs[1],
        f"{prefix}__dyn_fft_top3_freq": top_freqs[2],
        f"{prefix}__dyn_fft_top1_amp": top_amps[0],
        f"{prefix}__dyn_fft_top2_amp": top_amps[1],
        f"{prefix}__dyn_fft_top3_amp": top_amps[2],
        f"{prefix}__dyn_spectral_centroid": centroid,
        f"{prefix}__dyn_spectral_bandwidth": bandwidth,
        f"{prefix}__dyn_spectral_entropy": entropy,
        f"{prefix}__dyn_spectral_rolloff_85": rolloff,
    }
    feature_map.update(
        _band_ratio_features(
            prefix=prefix,
            spectrum_info=spectrum_info,
            bands=SECONDARY_FREQ_BANDS,
        )
    )
    return feature_map


def zero_crossing_rate(signal: np.ndarray) -> float:
    if signal.size < 2:
        return 0.0
    signs = np.sign(signal)
    signs[signs == 0] = 1
    return float(np.count_nonzero(signs[1:] != signs[:-1]) / (signal.size - 1))


def build_cross_channel_features(channel_caches: dict[str, dict[str, object]]) -> dict[str, float]:
    acc_channels = [cache for cache in channel_caches.values() if cache["group"] == "acc"]
    strain_channels = [cache for cache in channel_caches.values() if cache["group"] == "strain"]
    feature_map: dict[str, float] = {}
    feature_map.update(_group_stat_features(acc_channels, prefix="acc"))
    feature_map.update(_group_stat_features(strain_channels, prefix="strain"))
    feature_map["acc_energy_to_strain_energy"] = _safe_ratio(
        feature_map["acc_energy_sum"], feature_map["strain_energy_sum"]
    )
    feature_map["acc_rms_mean_to_strain_rms_mean"] = _safe_ratio(
        feature_map["acc_rms_mean"], feature_map["strain_rms_mean"]
    )
    return feature_map


def _group_stat_features(channels: list[dict[str, object]], prefix: str) -> dict[str, float]:
    rms_values = np.array([float(channel["rms"]) for channel in channels], dtype=float)
    peak_freq_values = np.array([float(channel["peak_freq"]) for channel in channels], dtype=float)
    energy_values = np.array([float(channel["energy"]) for channel in channels], dtype=float)
    corr_values = _group_correlations(channels)
    return {
        f"{prefix}_rms_mean": _mean(rms_values),
        f"{prefix}_rms_std": _std(rms_values),
        f"{prefix}_rms_range": _range(rms_values),
        f"{prefix}_peak_freq_mean": _mean(peak_freq_values),
        f"{prefix}_peak_freq_std": _std(peak_freq_values),
        f"{prefix}_peak_freq_range": _range(peak_freq_values),
        f"{prefix}_energy_sum": float(np.sum(energy_values)) if len(energy_values) else 0.0,
        f"{prefix}_energy_mean": _mean(energy_values),
        f"{prefix}_energy_std": _std(energy_values),
        f"{prefix}_corr_mean": _mean(corr_values),
        f"{prefix}_corr_std": _std(corr_values),
        f"{prefix}_corr_max_abs": float(np.max(np.abs(corr_values))) if len(corr_values) else 0.0,
    }


def _group_correlations(channels: list[dict[str, object]]) -> np.ndarray:
    if len(channels) < 2:
        return np.array([], dtype=float)
    corr_values: list[float] = []
    for first_index in range(len(channels) - 1):
        x = np.asarray(channels[first_index]["dynamic_signal"], dtype=float)
        for second_index in range(first_index + 1, len(channels)):
            y = np.asarray(channels[second_index]["dynamic_signal"], dtype=float)
            if np.std(x, ddof=0) == 0 or np.std(y, ddof=0) == 0:
                corr = 0.0
            else:
                corr = float(np.corrcoef(x, y)[0, 1])
                if np.isnan(corr):
                    corr = 0.0
            corr_values.append(corr)
    return np.array(corr_values, dtype=float)


def get_channel_group(column_name: str) -> str:
    if ".Acc" in column_name:
        return "acc"
    if column_name.startswith("应变传感器"):
        return "strain"
    return "other"


def get_group_feature_columns(feature_df: pd.DataFrame, group_name: str) -> list[str]:
    spec = GROUP_SPEC_MAP[group_name]
    signal_prefixes = sorted(
        {
            column.split("__", maxsplit=1)[0]
            for column in feature_df.columns
            if "__" in column and not column.startswith("__")
        }
    )
    feature_columns: list[str] = []
    for prefix in signal_prefixes:
        feature_columns.extend(f"{prefix}__{suffix}" for suffix in BASELINE_SUFFIXES)
        feature_columns.extend(f"{prefix}__{suffix}" for suffix in BASE_DYNAMIC_SUFFIXES)
        if spec.include_robust_time:
            feature_columns.extend(f"{prefix}__{suffix}" for suffix in ROBUST_SUFFIXES)
        if spec.include_freq_shape:
            feature_columns.extend(f"{prefix}__{suffix}" for suffix in FREQ_SHAPE_SUFFIXES)

    if spec.include_cross_channel:
        feature_columns.extend(CROSS_CHANNEL_COLUMNS)
    if spec.include_quality:
        feature_columns.extend(QUALITY_FEATURE_COLUMNS)
    return [column for column in feature_columns if column in feature_df.columns]


def build_feature_manifest(feature_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for spec in GROUP_SPECS:
        feature_columns = get_group_feature_columns(feature_df, spec.group_name)
        for order, column in enumerate(feature_columns):
            rows.append(
                {
                    "group_name": spec.group_name,
                    "feature_order": order,
                    "feature_column": column,
                }
            )
    return pd.DataFrame(rows)


def build_loco_split_map(labeled_df: pd.DataFrame) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    case_ids = sorted(labeled_df["case_id"].unique())
    split_map: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    case_id_values = labeled_df["case_id"].to_numpy(dtype=int, copy=False)
    for case_id in case_ids:
        valid_idx = np.flatnonzero(case_id_values == case_id)
        train_idx = np.flatnonzero(case_id_values != case_id)
        split_map[case_id] = (train_idx, valid_idx)
    return split_map


def build_group_matrix_cache(
    feature_df: pd.DataFrame,
    include_rpm_aware: bool,
) -> dict[tuple[str, str], np.ndarray]:
    labeled_df = feature_df[feature_df["wind_speed"].notna()].copy()
    cache: dict[tuple[str, str], np.ndarray] = {}
    for spec in GROUP_SPECS:
        rpm_free_columns = get_group_feature_columns(feature_df, spec.group_name)
        cache[(spec.group_name, "rpm_free")] = labeled_df[rpm_free_columns].to_numpy(dtype=float, copy=False)
        if include_rpm_aware:
            cache[(spec.group_name, "rpm_aware")] = labeled_df[[*rpm_free_columns, "rpm"]].to_numpy(
                dtype=float,
                copy=False,
            )
    return cache


def run_screening_round(
    feature_df: pd.DataFrame,
    runtime_config: Phase1RuntimeConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    labeled_df = feature_df[feature_df["wind_speed"].notna()].copy()
    split_map = build_loco_split_map(labeled_df)
    matrix_cache = build_group_matrix_cache(feature_df, include_rpm_aware=False)
    summary_rows: list[dict[str, object]] = []
    case_frames: list[pd.DataFrame] = []
    if runtime_config.max_workers > 1:
        with ThreadPoolExecutor(max_workers=runtime_config.max_workers) as executor:
            futures = [
                executor.submit(
                    _evaluate_group_task,
                    labeled_df,
                    split_map,
                    matrix_cache[(spec.group_name, "rpm_free")],
                    spec.group_name,
                    "Ridge",
                    len(get_group_feature_columns(labeled_df, spec.group_name)),
                    runtime_config,
                )
                for spec in GROUP_SPECS
            ]
            for future in futures:
                summary_row, case_frame = future.result()
                summary_rows.append(summary_row)
                case_frames.append(case_frame)
    else:
        for spec in GROUP_SPECS:
            summary_row, case_frame = _evaluate_group_task(
                labeled_df=labeled_df,
                split_map=split_map,
                matrix=matrix_cache[(spec.group_name, "rpm_free")],
                feature_set_name=spec.group_name,
                model_name="Ridge",
                feature_count=len(get_group_feature_columns(labeled_df, spec.group_name)),
                runtime_config=runtime_config,
            )
            summary_rows.append(summary_row)
            case_frames.append(case_frame)
    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["case_mae", "case_rmse", "feature_count", "group_name"]
    ).reset_index(drop=True)
    case_level_df = pd.concat(case_frames, ignore_index=True)
    base_case_errors = (
        case_level_df[case_level_df["feature_set"] == "G0_BASE"][["case_id", "abs_error"]]
        .rename(columns={"abs_error": "base_abs_error"})
    )
    case_level_df = case_level_df.merge(base_case_errors, on="case_id", how="left")
    case_level_df["abs_error_delta_vs_base"] = case_level_df["abs_error"] - case_level_df["base_abs_error"]
    return summary_df, case_level_df


def select_finalists(screening_summary: pd.DataFrame) -> pd.DataFrame:
    base_row = screening_summary.loc[screening_summary["group_name"] == "G0_BASE"].iloc[0]
    candidates = screening_summary.copy()
    candidates["mae_improvement_vs_base"] = float(base_row["case_mae"]) - candidates["case_mae"]
    candidates["rmse_delta_vs_base"] = candidates["case_rmse"] - float(base_row["case_rmse"])
    candidates = candidates[
        (candidates["group_name"] != "G0_BASE")
        & (candidates["mae_improvement_vs_base"] >= 0.01)
        & (candidates["rmse_delta_vs_base"] <= 0)
    ].copy()
    if candidates.empty:
        return candidates
    return candidates.sort_values(
        ["case_mae", "case_rmse", "feature_count", "group_name"]
    ).head(2).reset_index(drop=True)


def run_finalist_round(
    feature_df: pd.DataFrame,
    finalists: pd.DataFrame,
    runtime_config: Phase1RuntimeConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    labeled_df = feature_df[feature_df["wind_speed"].notna()].copy()
    split_map = build_loco_split_map(labeled_df)
    matrix_cache = build_group_matrix_cache(feature_df, include_rpm_aware=True)
    summary_rows: list[dict[str, object]] = []
    case_frames: list[pd.DataFrame] = []
    task_payloads: list[tuple[str, str, str, int]] = []
    for _, finalist in finalists.iterrows():
        group_name = str(finalist["group_name"])
        feature_count = len(get_group_feature_columns(feature_df, group_name))
        task_payloads.extend(
            [
                (group_name, "rpm_free", "Ridge", feature_count),
                (group_name, "rpm_free", "RandomForestRegressor", feature_count),
                (group_name, "rpm_free", "HistGradientBoostingRegressor", feature_count),
                (group_name, "rpm_aware", "Ridge", feature_count + 1),
                (group_name, "rpm_aware", "RandomForestRegressor", feature_count + 1),
                (group_name, "rpm_aware", "HistGradientBoostingRegressor", feature_count + 1),
            ]
        )

    if runtime_config.max_workers > 1 and task_payloads:
        with ThreadPoolExecutor(max_workers=runtime_config.max_workers) as executor:
            futures = [
                executor.submit(
                    _evaluate_group_task,
                    labeled_df,
                    split_map,
                    matrix_cache[(group_name, task_mode)],
                    f"{group_name}__{task_mode}",
                    model_name,
                    feature_count,
                    runtime_config,
                    group_name,
                    task_mode,
                )
                for group_name, task_mode, model_name, feature_count in task_payloads
            ]
            for future in futures:
                summary_row, case_frame = future.result()
                summary_rows.append(summary_row)
                case_frames.append(case_frame)
    else:
        for group_name, task_mode, model_name, feature_count in task_payloads:
            summary_row, case_frame = _evaluate_group_task(
                labeled_df=labeled_df,
                split_map=split_map,
                matrix=matrix_cache[(group_name, task_mode)],
                feature_set_name=f"{group_name}__{task_mode}",
                model_name=model_name,
                feature_count=feature_count,
                runtime_config=runtime_config,
                group_name=group_name,
                task_mode=task_mode,
            )
            summary_rows.append(summary_row)
            case_frames.append(case_frame)
    summary_df = pd.DataFrame(summary_rows)
    if summary_df.empty:
        return summary_df, pd.DataFrame()
    summary_df = summary_df.sort_values(
        ["group_name", "task_mode", "case_mae", "case_rmse", "model_name"]
    ).reset_index(drop=True)
    return summary_df, pd.concat(case_frames, ignore_index=True)


def evaluate_loco(
    labeled_df: pd.DataFrame,
    split_map: dict[int, tuple[np.ndarray, np.ndarray]],
    matrix: np.ndarray,
    model_name: str,
    runtime_config: Phase1RuntimeConfig,
) -> pd.DataFrame:
    predictions: list[pd.DataFrame] = []
    y_all = labeled_df["wind_speed"].to_numpy(dtype=float, copy=False)
    for case_id, (train_idx, valid_idx) in split_map.items():
        estimator = build_estimator(model_name, runtime_config)
        estimator.fit(matrix[train_idx], y_all[train_idx])
        pred = estimator.predict(matrix[valid_idx])
        valid_df = labeled_df.iloc[valid_idx]
        fold_df = valid_df[
            ["case_id", "file_name", "window_index", "start_time", "end_time", "wind_speed"]
        ].copy()
        fold_df = fold_df.rename(columns={"wind_speed": "true_wind_speed"})
        fold_df["pred_wind_speed"] = pred
        predictions.append(fold_df)
    return pd.concat(predictions, ignore_index=True)


def summarize_case_predictions(
    prediction_frame: pd.DataFrame,
    model_name: str,
    feature_set: str,
) -> pd.DataFrame:
    case_frame = (
        prediction_frame.groupby(["case_id", "file_name", "true_wind_speed"], as_index=False)["pred_wind_speed"]
        .mean()
        .rename(columns={"pred_wind_speed": "pred_mean"})
    )
    case_frame["abs_error"] = (case_frame["pred_mean"] - case_frame["true_wind_speed"]).abs()
    case_frame["model_name"] = model_name
    case_frame["feature_set"] = feature_set
    return case_frame


def build_estimator(model_name: str, runtime_config: Phase1RuntimeConfig):
    if model_name == "Ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    if model_name == "RandomForestRegressor":
        return RandomForestRegressor(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=runtime_config.rf_n_jobs,
        )
    if model_name == "HistGradientBoostingRegressor":
        return HistGradientBoostingRegressor(
            max_depth=4,
            learning_rate=0.05,
            max_iter=300,
            l2_regularization=0.1,
            random_state=42,
        )
    raise ValueError(f"未知模型: {model_name}")


def _evaluate_group_task(
    labeled_df: pd.DataFrame,
    split_map: dict[int, tuple[np.ndarray, np.ndarray]],
    matrix: np.ndarray,
    feature_set_name: str,
    model_name: str,
    feature_count: int,
    runtime_config: Phase1RuntimeConfig,
    group_name: str | None = None,
    task_mode: str | None = None,
) -> tuple[dict[str, object], pd.DataFrame]:
    prediction_frame = evaluate_loco(
        labeled_df=labeled_df,
        split_map=split_map,
        matrix=matrix,
        model_name=model_name,
        runtime_config=runtime_config,
    )
    case_frame = summarize_case_predictions(
        prediction_frame,
        model_name=model_name,
        feature_set=feature_set_name,
    )
    case_frame["feature_count"] = feature_count
    if group_name is not None:
        case_frame["group_name"] = group_name
    if task_mode is not None:
        case_frame["task_mode"] = task_mode
    errors = case_frame["pred_mean"] - case_frame["true_wind_speed"]
    summary_row: dict[str, object] = {
        "feature_count": feature_count,
        "case_mae": float(np.mean(np.abs(errors))),
        "case_rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "case_mape": float(
            np.mean(np.abs(errors) / case_frame["true_wind_speed"].to_numpy(dtype=float)) * 100
        ),
    }
    if group_name is None:
        summary_row["group_name"] = feature_set_name
    else:
        summary_row["group_name"] = group_name
        summary_row["task_mode"] = task_mode
        summary_row["model_name"] = model_name
    return summary_row, case_frame


def evaluate_promotion(
    screening_summary: pd.DataFrame,
    screening_cases: pd.DataFrame,
    finalists: pd.DataFrame,
    finalist_summary: pd.DataFrame,
    unlabeled_prediction: float | None,
) -> PromotionDecision:
    fail_reasons: list[str] = []
    if finalists.empty:
        return PromotionDecision(promoted=False, fail_reasons=["没有满足 finalist 条件的特征组。"])

    first_group = str(finalists.iloc[0]["group_name"])
    base_mae = float(screening_summary.loc[screening_summary["group_name"] == "G0_BASE", "case_mae"].iloc[0])
    first_mae = float(screening_summary.loc[screening_summary["group_name"] == first_group, "case_mae"].iloc[0])
    if base_mae - first_mae < 0.03:
        fail_reasons.append("第一轮 case_mae 相比 G0_BASE 的下降不足 0.03。")

    candidate_case_df = screening_cases[screening_cases["feature_set"] == first_group].copy()
    degraded_cases = int((candidate_case_df["abs_error_delta_vs_base"] > 0.10).sum())
    if degraded_cases > 3:
        fail_reasons.append(f"第一轮恶化超过 0.10 m/s 的工况数为 {degraded_cases}，超过上限 3。")

    rpm_free_rows = finalist_summary[
        (finalist_summary["group_name"] == first_group)
        & (finalist_summary["task_mode"] == "rpm_free")
    ].copy()
    if rpm_free_rows.empty:
        fail_reasons.append("第二轮未生成 rpm-free 比较结果。")
    else:
        best_rpm_free = rpm_free_rows.sort_values(["case_mae", "case_rmse", "model_name"]).iloc[0]
        ridge_row = rpm_free_rows[rpm_free_rows["model_name"] == "Ridge"].iloc[0]
        if (
            best_rpm_free["model_name"] != "Ridge"
            and float(ridge_row["case_mae"]) - float(best_rpm_free["case_mae"]) > 0.01
        ):
            fail_reasons.append("第二轮 rpm-free 中 Ridge 与最优模型的 case_mae 差距超过 0.01。")

    if unlabeled_prediction is None or np.isnan(unlabeled_prediction):
        fail_reasons.append("无标签工况未能生成 rpm-free 回退预测。")

    return PromotionDecision(promoted=len(fail_reasons) == 0, fail_reasons=fail_reasons)


def predict_unlabeled_with_group(
    feature_df: pd.DataFrame,
    group_name: str,
    runtime_config: Phase1RuntimeConfig,
) -> pd.DataFrame:
    labeled_df = feature_df[feature_df["wind_speed"].notna()].copy()
    unlabeled_df = feature_df[feature_df["wind_speed"].isna()].copy()
    feature_columns = get_group_feature_columns(feature_df, group_name)
    if unlabeled_df.empty:
        return pd.DataFrame(columns=["case_id", "file_name", "predicted_wind_speed"])
    estimator = build_estimator("Ridge", runtime_config)
    estimator.fit(labeled_df[feature_columns], labeled_df["wind_speed"])
    pred = estimator.predict(unlabeled_df[feature_columns])
    prediction_df = unlabeled_df[["case_id", "file_name"]].copy()
    prediction_df["pred_wind_speed"] = pred
    return (
        prediction_df.groupby(["case_id", "file_name"], as_index=False)["pred_wind_speed"]
        .mean()
        .rename(columns={"pred_wind_speed": "predicted_wind_speed"})
    )


def write_summary_markdown(
    output_dir: Path,
    screening_summary: pd.DataFrame,
    finalists: pd.DataFrame,
    finalist_summary: pd.DataFrame,
    promotion: PromotionDecision,
    unlabeled_predictions: pd.DataFrame,
) -> None:
    champion = screening_summary.iloc[0]
    lines = [
        "# 第一阶段特征组筛选结论",
        "",
        f"- 第一轮冠军：`{champion['group_name']}`",
        f"- 第一轮冠军 case_mae：`{float(champion['case_mae']):.4f}`",
        f"- 是否晋升：`{'yes' if promotion.promoted else 'no'}`",
        "",
        "## 第一轮 finalist",
        "",
    ]
    if finalists.empty:
        lines.append("- 无 finalist")
    else:
        for _, row in finalists.iterrows():
            lines.append(
                f"- `{row['group_name']}`: case_mae=`{float(row['case_mae']):.4f}`, case_rmse=`{float(row['case_rmse']):.4f}`, feature_count=`{int(row['feature_count'])}`"
            )

    lines.extend(["", "## 第二轮要点", ""])
    if finalist_summary.empty:
        lines.append("- 无第二轮结果")
    else:
        for group_name in finalist_summary["group_name"].drop_duplicates():
            group_rows = finalist_summary[
                (finalist_summary["group_name"] == group_name)
                & (finalist_summary["task_mode"] == "rpm_free")
            ].sort_values(["case_mae", "case_rmse", "model_name"])
            if group_rows.empty:
                continue
            best_row = group_rows.iloc[0]
            lines.append(
                f"- `{group_name}` rpm-free 最优：`{best_row['model_name']}`，case_mae=`{float(best_row['case_mae']):.4f}`"
            )

    lines.extend(["", "## 无标签回退推理", ""])
    if unlabeled_predictions.empty:
        lines.append("- 无无标签工况")
    else:
        for _, row in unlabeled_predictions.iterrows():
            lines.append(
                f"- `{row['file_name']}` 预测风速：`{float(row['predicted_wind_speed']):.4f} m/s`"
            )

    lines.extend(["", "## 晋升判定", ""])
    if promotion.promoted:
        lines.append("- 通过全部晋升门槛，可进入 `src/current/`。")
    else:
        for reason in promotion.fail_reasons:
            lines.append(f"- {reason}")

    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def create_case_mae_bar(screening_summary: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(screening_summary["group_name"], screening_summary["case_mae"], color="#4c78a8")
    ax.set_title("Phase 1 Screening Case MAE")
    ax.set_xlabel("Feature Group")
    ax.set_ylabel("Case MAE")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def create_case_delta_heatmap(case_level_df: pd.DataFrame, output_path: Path) -> None:
    plot_df = case_level_df.pivot(
        index="feature_set",
        columns="case_id",
        values="abs_error_delta_vs_base",
    ).sort_index()
    fig, ax = plt.subplots(figsize=(12, 5))
    image = ax.imshow(plot_df.to_numpy(dtype=float), cmap="coolwarm", aspect="auto")
    ax.set_title("Abs Error Delta vs G0_BASE")
    ax.set_xlabel("Case ID")
    ax.set_ylabel("Feature Group")
    ax.set_xticks(range(len(plot_df.columns)))
    ax.set_xticklabels(plot_df.columns.tolist())
    ax.set_yticks(range(len(plot_df.index)))
    ax.set_yticklabels(plot_df.index.tolist())
    fig.colorbar(image, ax=ax, fraction=0.03, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _compute_spectrum_info(signal: np.ndarray, sampling_rate: float) -> dict[str, object]:
    spectrum = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(signal.size, d=1.0 / sampling_rate)
    magnitudes = np.abs(spectrum)
    power = np.square(magnitudes)
    if len(freqs) <= 1:
        return {
            "freqs": np.array([], dtype=float),
            "magnitudes": np.array([], dtype=float),
            "power": np.array([], dtype=float),
            "peak_freq": 0.0,
            "peak_amp": 0.0,
            "total_energy": 0.0,
            "top_freqs": [0.0, 0.0, 0.0],
            "top_amps": [0.0, 0.0, 0.0],
        }

    non_dc_freqs = freqs[1:]
    non_dc_magnitudes = magnitudes[1:]
    non_dc_power = power[1:]
    total_energy = float(np.sum(non_dc_power))
    if total_energy <= 0.0 or len(non_dc_magnitudes) == 0:
        return {
            "freqs": non_dc_freqs,
            "magnitudes": non_dc_magnitudes,
            "power": non_dc_power,
            "peak_freq": 0.0,
            "peak_amp": 0.0,
            "total_energy": 0.0,
            "top_freqs": [0.0, 0.0, 0.0],
            "top_amps": [0.0, 0.0, 0.0],
        }

    peak_idx = int(np.argmax(non_dc_magnitudes))
    order = np.argsort(non_dc_magnitudes)[::-1]
    top_freqs = [0.0, 0.0, 0.0]
    top_amps = [0.0, 0.0, 0.0]
    for index, source_idx in enumerate(order[:3]):
        top_freqs[index] = float(non_dc_freqs[source_idx])
        top_amps[index] = float(non_dc_magnitudes[source_idx])
    return {
        "freqs": non_dc_freqs,
        "magnitudes": non_dc_magnitudes,
        "power": non_dc_power,
        "peak_freq": float(non_dc_freqs[peak_idx]),
        "peak_amp": float(non_dc_magnitudes[peak_idx]),
        "total_energy": total_energy,
        "top_freqs": top_freqs,
        "top_amps": top_amps,
    }


def _band_ratio_features(
    prefix: str,
    spectrum_info: dict[str, object],
    bands: Iterable[tuple[float, float]],
) -> dict[str, float]:
    freqs = np.asarray(spectrum_info["freqs"], dtype=float)
    power = np.asarray(spectrum_info["power"], dtype=float)
    total_energy = float(spectrum_info["total_energy"])
    feature_map: dict[str, float] = {}
    for low, high in bands:
        mask = (freqs >= low) & (freqs < high)
        band_energy = float(np.sum(power[mask])) if len(power) else 0.0
        ratio = band_energy / total_energy if total_energy > 0 else 0.0
        band_name = f"{int(low)}_{int(high)}hz"
        feature_map[f"{prefix}__dyn_fft_band_ratio_{band_name}"] = ratio
    return feature_map


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def _mean(values: np.ndarray) -> float:
    return float(np.mean(values)) if len(values) else 0.0


def _std(values: np.ndarray) -> float:
    return float(np.std(values, ddof=0)) if len(values) else 0.0


def _range(values: np.ndarray) -> float:
    return float(np.max(values) - np.min(values)) if len(values) else 0.0
