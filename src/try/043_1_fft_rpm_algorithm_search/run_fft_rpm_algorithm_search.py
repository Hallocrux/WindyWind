from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.current.data_loading import DatasetRecord, get_common_signal_columns, load_clean_signal_frame, scan_dataset_records  # noqa: E402

TRY_NAME = "043_1_fft_rpm_algorithm_search"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "try" / TRY_NAME
ADDED_MANIFEST_PATH = REPO_ROOT / "data" / "added" / "dataset_manifest.csv"
ADDED_STANDARD_DIR = REPO_ROOT / "data" / "added" / "standardized_datasets"
ACC_SUFFIXES = (".AccX", ".AccY", ".AccZ")
SEARCH_LOW_HZ = 1.2
SEARCH_HIGH_HZ = 4.8
MAX_ANALYSIS_HZ = 15.0
SEARCH_STEP_HZ = 0.01
EPS = 1e-8


@dataclass(frozen=True)
class VariantSpec:
    variant_name: str
    estimator: str
    spectrum_mode: str
    spectral_window_seconds: float | None = None
    window_estimator_seconds: float | None = None
    window_step_ratio: float = 0.5
    harmonic_weights: tuple[float, ...] = (0.7, 0.9, 1.0, 0.6)
    top_k_channels: int = 6
    peak_count: int = 8
    use_autocorr_prior: bool = False
    autocorr_weight: float = 0.0
    search_low_hz: float = SEARCH_LOW_HZ
    search_high_hz: float = SEARCH_HIGH_HZ
    max_analysis_hz: float = MAX_ANALYSIS_HZ


@dataclass(frozen=True)
class EstimateResult:
    pred_rpm: float
    pred_hz: float
    confidence: float
    score: float
    runner_up_rpm: float
    runner_up_gap_rpm: float
    dominant_harmonic: int
    selected_channel_count: int
    segment_count: int
    window_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="搜索 FFT 方向的最佳解析 RPM 算法。")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=None,
        help="只运行指定变体，默认运行全部内置变体。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    variants = build_variant_specs()
    if args.variants is not None:
        requested = set(args.variants)
        variants = [spec for spec in variants if spec.variant_name in requested]
        missing = sorted(requested - {spec.variant_name for spec in variants})
        if missing:
            raise ValueError(f"未知变体: {missing}")

    final_records = [record for record in scan_dataset_records() if record.is_labeled]
    added_records = load_added_records()
    all_records = [*final_records, *added_records]
    common_signal_columns = get_common_signal_columns(all_records)
    cleaned_frames = {
        record.case_id: load_clean_signal_frame(record, common_signal_columns)
        for record in all_records
    }

    variant_rows: list[dict[str, object]] = []
    case_rows: list[dict[str, object]] = []
    for spec in variants:
        variant_rows.append(spec_to_row(spec))
        for domain, records in (
            ("final_direct", final_records),
            ("added_external", added_records),
        ):
            for record in records:
                estimate = estimate_record_rpm(
                    record=record,
                    frame=cleaned_frames[record.case_id],
                    spec=spec,
                )
                case_rows.append(
                    {
                        "domain": domain,
                        "variant_name": spec.variant_name,
                        "case_id": record.case_id,
                        "file_name": record.file_name,
                        "true_rpm": float(record.rpm),
                        "pred_rpm": estimate.pred_rpm,
                        "signed_error": estimate.pred_rpm - float(record.rpm),
                        "abs_error": abs(estimate.pred_rpm - float(record.rpm)),
                        "confidence": estimate.confidence,
                        "score": estimate.score,
                        "runner_up_rpm": estimate.runner_up_rpm,
                        "runner_up_gap_rpm": estimate.runner_up_gap_rpm,
                        "dominant_harmonic": estimate.dominant_harmonic,
                        "selected_channel_count": estimate.selected_channel_count,
                        "segment_count": estimate.segment_count,
                        "window_count": estimate.window_count,
                    }
                )

    variant_df = pd.DataFrame(variant_rows).sort_values("variant_name").reset_index(drop=True)
    case_df = pd.DataFrame(case_rows).sort_values(
        ["domain", "variant_name", "case_id"]
    ).reset_index(drop=True)
    hybrid_variant_df, hybrid_case_df = build_hybrid_peak_variants(case_df)
    if not hybrid_variant_df.empty:
        variant_df = pd.concat([variant_df, hybrid_variant_df], ignore_index=True)
        variant_df = variant_df.sort_values("variant_name").reset_index(drop=True)
        case_df = pd.concat([case_df, hybrid_case_df], ignore_index=True)
        case_df = case_df.sort_values(["domain", "variant_name", "case_id"]).reset_index(drop=True)
    summary_df = build_summary(case_df)
    best_df = build_best_variant_by_domain(summary_df)
    failure_df = build_failure_cases(case_df, best_df)
    write_summary_markdown(args.output_dir / "summary.md", summary_df, best_df, failure_df)

    variant_df.to_csv(args.output_dir / "variant_config_table.csv", index=False, encoding="utf-8-sig")
    case_df.to_csv(args.output_dir / "case_level_predictions.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(args.output_dir / "summary.csv", index=False, encoding="utf-8-sig")
    best_df.to_csv(args.output_dir / "best_variant_by_domain.csv", index=False, encoding="utf-8-sig")
    failure_df.to_csv(args.output_dir / "failure_cases.csv", index=False, encoding="utf-8-sig")

    best_added = best_df.loc[best_df["domain"] == "added_external"].iloc[0]
    print("043_1 FFT RPM 算法搜索完成。")
    print(f"输出目录: {args.output_dir}")
    print(
        "best added: "
        f"{best_added['variant_name']} | case_mae={best_added['case_mae']:.4f} | "
        f"case_rmse={best_added['case_rmse']:.4f}"
    )


def build_variant_specs() -> list[VariantSpec]:
    return [
        VariantSpec("fft_peak_1x_whole", estimator="peak_1x", spectrum_mode="whole"),
        VariantSpec(
            "fft_peak_1x_welch_8s",
            estimator="peak_1x",
            spectrum_mode="welch",
            spectral_window_seconds=8.0,
        ),
        VariantSpec(
            "fft_peak_1x_welch_12s",
            estimator="peak_1x",
            spectrum_mode="welch",
            spectral_window_seconds=12.0,
        ),
        VariantSpec("fft_peak_3x_whole", estimator="peak_3x", spectrum_mode="whole"),
        VariantSpec(
            "window_peak_1x_conf_8s",
            estimator="peak_1x",
            spectrum_mode="whole",
            window_estimator_seconds=8.0,
        ),
        VariantSpec(
            "window_peak_1x_conf_12s",
            estimator="peak_1x",
            spectrum_mode="whole",
            window_estimator_seconds=12.0,
        ),
        VariantSpec("fft_vote_whole", estimator="vote_backprojection", spectrum_mode="whole"),
        VariantSpec("harmonic_template_whole", estimator="harmonic_template", spectrum_mode="whole"),
        VariantSpec(
            "harmonic_template_autocorr_whole",
            estimator="harmonic_template",
            spectrum_mode="whole",
            use_autocorr_prior=True,
            autocorr_weight=0.35,
        ),
        VariantSpec(
            "harmonic_template_welch_12s",
            estimator="harmonic_template",
            spectrum_mode="welch",
            spectral_window_seconds=12.0,
        ),
        VariantSpec(
            "harmonic_template_autocorr_welch_12s",
            estimator="harmonic_template",
            spectrum_mode="welch",
            spectral_window_seconds=12.0,
            use_autocorr_prior=True,
            autocorr_weight=0.35,
        ),
        VariantSpec(
            "window_template_conf_12s",
            estimator="harmonic_template",
            spectrum_mode="whole",
            window_estimator_seconds=12.0,
        ),
        VariantSpec(
            "window_vote_conf_12s",
            estimator="vote_backprojection",
            spectrum_mode="whole",
            window_estimator_seconds=12.0,
        ),
        VariantSpec(
            "window_template_autocorr_conf_12s",
            estimator="harmonic_template",
            spectrum_mode="whole",
            window_estimator_seconds=12.0,
            use_autocorr_prior=True,
            autocorr_weight=0.35,
        ),
    ]


def build_hybrid_peak_variants(case_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    whole_name = "fft_peak_1x_whole"
    window_name = "window_peak_1x_conf_8s"
    if case_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    base_variants = set(case_df["variant_name"].unique())
    if whole_name not in base_variants or window_name not in base_variants:
        return pd.DataFrame(), pd.DataFrame()

    variant_name = "hybrid_peak_1x_whole_window8_gate150"
    whole_df = case_df.loc[case_df["variant_name"] == whole_name].copy()
    window_df = case_df.loc[case_df["variant_name"] == window_name].copy()
    merged = whole_df.merge(
        window_df,
        on=["domain", "case_id", "file_name", "true_rpm"],
        suffixes=("_whole", "_window8"),
    )

    hybrid_rows: list[dict[str, object]] = []
    for _, row in merged.iterrows():
        pred_whole = float(row["pred_rpm_whole"])
        pred_window = float(row["pred_rpm_window8"])
        if np.isclose(pred_whole, pred_window, atol=1e-9):
            use_window = False
        else:
            use_window = max(pred_whole, pred_window) < 150.0 or pred_window > pred_whole
        source = "window_peak_1x_conf_8s" if use_window else "fft_peak_1x_whole"
        suffix = "_window8" if use_window else "_whole"
        pred_rpm = float(row[f"pred_rpm{suffix}"])
        signed_error = pred_rpm - float(row["true_rpm"])
        hybrid_rows.append(
            {
                "domain": row["domain"],
                "variant_name": variant_name,
                "case_id": int(row["case_id"]),
                "file_name": row["file_name"],
                "true_rpm": float(row["true_rpm"]),
                "pred_rpm": pred_rpm,
                "signed_error": signed_error,
                "abs_error": abs(signed_error),
                "confidence": float(row[f"confidence{suffix}"]),
                "score": float(row[f"score{suffix}"]),
                "runner_up_rpm": float(row[f"runner_up_rpm{suffix}"]),
                "runner_up_gap_rpm": float(row[f"runner_up_gap_rpm{suffix}"]),
                "dominant_harmonic": int(row[f"dominant_harmonic{suffix}"]),
                "selected_channel_count": int(row[f"selected_channel_count{suffix}"]),
                "segment_count": int(row[f"segment_count{suffix}"]),
                "window_count": int(row[f"window_count{suffix}"]),
                "hybrid_source": source,
            }
        )

    variant_row = pd.DataFrame(
        [
            {
                "variant_name": variant_name,
                "estimator": "hybrid_gate150",
                "spectrum_mode": "whole+window8",
                "spectral_window_seconds": np.nan,
                "window_estimator_seconds": 8.0,
                "window_step_ratio": 0.5,
                "harmonic_weights": "0.70,0.90,1.00,0.60",
                "top_k_channels": 6,
                "peak_count": 8,
                "use_autocorr_prior": False,
                "autocorr_weight": 0.0,
                "search_low_hz": SEARCH_LOW_HZ,
                "search_high_hz": SEARCH_HIGH_HZ,
                "max_analysis_hz": MAX_ANALYSIS_HZ,
            }
        ]
    )
    return variant_row, pd.DataFrame(hybrid_rows)


def spec_to_row(spec: VariantSpec) -> dict[str, object]:
    row = asdict(spec)
    row["harmonic_weights"] = ",".join(f"{value:.2f}" for value in spec.harmonic_weights)
    return row


def load_added_records() -> list[DatasetRecord]:
    manifest_df = pd.read_csv(ADDED_MANIFEST_PATH)
    records: list[DatasetRecord] = []
    for _, row in manifest_df.iterrows():
        case_id = int(row["case_id"])
        records.append(
            DatasetRecord(
                case_id=case_id,
                display_name=str(row["display_name"]),
                file_name=f"工况{case_id}.csv",
                file_path=ADDED_STANDARD_DIR / f"工况{case_id}.csv",
                wind_speed=float(row["wind_speed"]) if not pd.isna(row["wind_speed"]) else None,
                rpm=float(row["rpm"]) if not pd.isna(row["rpm"]) else None,
                is_labeled=not pd.isna(row["wind_speed"]) and not pd.isna(row["rpm"]),
                original_file_name=str(row["original_file_name"]),
                label_source=str(row["label_source"]),
                notes=str(row["notes"]),
            )
        )
    return [record for record in records if record.is_labeled]


def estimate_record_rpm(record: DatasetRecord, frame: pd.DataFrame, spec: VariantSpec) -> EstimateResult:
    acc_columns = [
        column
        for column in frame.columns
        if column.startswith("WSMS") and column.endswith(ACC_SUFFIXES)
    ]
    if not acc_columns:
        raise ValueError(f"{record.file_name} 没有可用加速度通道。")

    segment_predictions: list[dict[str, float | int]] = []
    window_count = 0
    for _, segment_df in frame.groupby("__segment_id", sort=True):
        values = segment_df[acc_columns].to_numpy(dtype=float).T
        if values.shape[1] < 32:
            continue
        sampling_rate = estimate_sampling_rate(segment_df)
        if spec.window_estimator_seconds is None:
            local_estimate = estimate_signal_matrix(values, sampling_rate, spec)
            segment_predictions.append(
                {
                    "pred_rpm": local_estimate.pred_rpm,
                    "weight": max(values.shape[1] / sampling_rate, 1.0) * max(local_estimate.confidence, 1e-3),
                    "score": local_estimate.score,
                    "confidence": local_estimate.confidence,
                    "runner_up_rpm": local_estimate.runner_up_rpm,
                    "dominant_harmonic": local_estimate.dominant_harmonic,
                }
            )
            window_count += 1
            continue

        window_size = int(round(spec.window_estimator_seconds * sampling_rate))
        step_size = max(int(round(window_size * spec.window_step_ratio)), 1)
        if values.shape[1] < window_size:
            local_estimate = estimate_signal_matrix(values, sampling_rate, spec)
            segment_predictions.append(
                {
                    "pred_rpm": local_estimate.pred_rpm,
                    "weight": max(values.shape[1] / sampling_rate, 1.0) * max(local_estimate.confidence, 1e-3),
                    "score": local_estimate.score,
                    "confidence": local_estimate.confidence,
                    "runner_up_rpm": local_estimate.runner_up_rpm,
                    "dominant_harmonic": local_estimate.dominant_harmonic,
                }
            )
            window_count += 1
            continue

        local_rpms: list[float] = []
        local_weights: list[float] = []
        local_scores: list[float] = []
        local_harmonics: list[int] = []
        local_runner_ups: list[float] = []
        for start in range(0, values.shape[1] - window_size + 1, step_size):
            window_values = values[:, start : start + window_size]
            local_estimate = estimate_signal_matrix(window_values, sampling_rate, spec)
            local_rpms.append(local_estimate.pred_rpm)
            local_weights.append(max(local_estimate.confidence, 1e-3))
            local_scores.append(local_estimate.score)
            local_harmonics.append(local_estimate.dominant_harmonic)
            local_runner_ups.append(local_estimate.runner_up_rpm)
        if local_rpms:
            local_rpms_array = np.asarray(local_rpms, dtype=float)
            local_weights_array = np.asarray(local_weights, dtype=float)
            segment_pred = weighted_median(local_rpms_array, local_weights_array)
            segment_runner_up = weighted_runner_up(local_rpms_array, local_weights_array, segment_pred)
            segment_predictions.append(
                {
                    "pred_rpm": segment_pred,
                    "weight": float(np.sum(local_weights)),
                    "score": float(np.mean(local_scores)),
                    "confidence": confidence_from_candidates(
                        local_rpms_array,
                        local_weights_array,
                        segment_pred,
                        segment_runner_up,
                    ),
                    "runner_up_rpm": segment_runner_up,
                    "dominant_harmonic": dominant_vote(local_harmonics),
                }
            )
            window_count += len(local_rpms)

    if not segment_predictions:
        raise ValueError(f"{record.file_name} 无有效连续段可做 FFT RPM 估计。")

    rpm_values = np.asarray([float(item["pred_rpm"]) for item in segment_predictions], dtype=float)
    weights = np.asarray([float(item["weight"]) for item in segment_predictions], dtype=float)
    scores = np.asarray([float(item["score"]) for item in segment_predictions], dtype=float)
    segment_confidences = np.asarray(
        [float(item["confidence"]) for item in segment_predictions],
        dtype=float,
    )
    dominant_harmonics = [int(item["dominant_harmonic"]) for item in segment_predictions]
    pred_rpm = weighted_median(rpm_values, weights)
    runner_up_rpm = weighted_runner_up(rpm_values, weights, pred_rpm)
    if len(segment_predictions) == 1:
        confidence = float(segment_confidences[0])
        runner_up_rpm = float(segment_predictions[0]["runner_up_rpm"])
    else:
        confidence = confidence_from_candidates(rpm_values, weights, pred_rpm, runner_up_rpm)
    return EstimateResult(
        pred_rpm=float(pred_rpm),
        pred_hz=float(pred_rpm / 60.0),
        confidence=float(confidence),
        score=float(np.average(scores, weights=np.maximum(weights, EPS))),
        runner_up_rpm=float(runner_up_rpm),
        runner_up_gap_rpm=float(abs(pred_rpm - runner_up_rpm)),
        dominant_harmonic=dominant_vote(dominant_harmonics),
        selected_channel_count=min(spec.top_k_channels, len(acc_columns)),
        segment_count=len(segment_predictions),
        window_count=window_count,
    )


def estimate_signal_matrix(values: np.ndarray, sampling_rate: float, spec: VariantSpec) -> EstimateResult:
    freqs, amplitude_matrix, quality = build_spectrum_matrix(
        values=values,
        sampling_rate=sampling_rate,
        spectrum_mode=spec.spectrum_mode,
        spectral_window_seconds=spec.spectral_window_seconds,
        max_analysis_hz=spec.max_analysis_hz,
    )
    search_grid = np.arange(spec.search_low_hz, spec.search_high_hz + SEARCH_STEP_HZ / 2.0, SEARCH_STEP_HZ)
    selected_indices = select_top_quality_channels(quality, spec.top_k_channels)
    selected_amplitudes = amplitude_matrix[selected_indices]

    if spec.estimator == "peak_1x":
        score_grid = aggregate_peak_curve(search_grid, freqs, selected_amplitudes, harmonic=1)
        dominant_harmonic = 1
    elif spec.estimator == "peak_3x":
        score_grid = aggregate_peak_curve(search_grid, freqs, selected_amplitudes, harmonic=3)
        dominant_harmonic = 3
    elif spec.estimator == "vote_backprojection":
        score_grid = build_vote_backprojection_curve(
            search_grid=search_grid,
            freqs=freqs,
            amplitudes=selected_amplitudes,
            harmonic_weights=spec.harmonic_weights,
            peak_count=spec.peak_count,
        )
        dominant_harmonic = dominant_harmonic_from_curve(
            search_grid=search_grid,
            freqs=freqs,
            amplitudes=selected_amplitudes,
            harmonic_weights=spec.harmonic_weights,
            score_grid=score_grid,
        )
    elif spec.estimator == "harmonic_template":
        score_grid = build_harmonic_template_curve(
            search_grid=search_grid,
            freqs=freqs,
            amplitudes=selected_amplitudes,
            harmonic_weights=spec.harmonic_weights,
        )
        dominant_harmonic = dominant_harmonic_from_curve(
            search_grid=search_grid,
            freqs=freqs,
            amplitudes=selected_amplitudes,
            harmonic_weights=spec.harmonic_weights,
            score_grid=score_grid,
        )
    else:
        raise ValueError(f"未知 estimator: {spec.estimator}")

    raw_score_grid = score_grid.copy()
    if spec.use_autocorr_prior:
        autocorr_grid = build_autocorr_prior(search_grid, values[selected_indices], sampling_rate)
        score_grid = normalize_curve(score_grid) + spec.autocorr_weight * normalize_curve(autocorr_grid)
    else:
        score_grid = normalize_curve(score_grid)

    best_index = int(np.argmax(score_grid))
    pred_hz = float(search_grid[best_index])
    second_index = runner_up_index(score_grid, best_index)
    runner_up_hz = float(search_grid[second_index])
    best_raw_score = float(raw_score_grid[best_index])
    baseline_score = float(np.median(raw_score_grid))
    return EstimateResult(
        pred_rpm=pred_hz * 60.0,
        pred_hz=pred_hz,
        confidence=float(local_confidence(score_grid, best_index)),
        score=float(best_raw_score / max(baseline_score, EPS)),
        runner_up_rpm=runner_up_hz * 60.0,
        runner_up_gap_rpm=abs(pred_hz - runner_up_hz) * 60.0,
        dominant_harmonic=dominant_harmonic,
        selected_channel_count=len(selected_indices),
        segment_count=1,
        window_count=1,
    )


def build_spectrum_matrix(
    *,
    values: np.ndarray,
    sampling_rate: float,
    spectrum_mode: str,
    spectral_window_seconds: float | None,
    max_analysis_hz: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    amplitude_rows: list[np.ndarray] = []
    quality_rows: list[float] = []
    freqs: np.ndarray | None = None
    for channel_values in values:
        centered = channel_values - np.mean(channel_values)
        if spectrum_mode == "welch":
            spectrum_freqs, channel_amplitudes = compute_welch_amplitude(
                centered,
                sampling_rate,
                window_seconds=spectral_window_seconds,
            )
        else:
            spectrum_freqs, channel_amplitudes = compute_single_sided_amplitude(centered, sampling_rate)
        mask = spectrum_freqs <= max_analysis_hz
        spectrum_freqs = spectrum_freqs[mask]
        channel_amplitudes = channel_amplitudes[mask]
        band_mask = (spectrum_freqs >= 0.5) & (spectrum_freqs <= max_analysis_hz)
        band_amplitudes = channel_amplitudes[band_mask]
        if band_amplitudes.size == 0:
            normalized = np.zeros_like(channel_amplitudes)
            quality = 0.0
        else:
            max_amp = float(np.max(band_amplitudes))
            median_amp = float(np.median(band_amplitudes))
            normalized = channel_amplitudes / max(max_amp, EPS)
            quality = max_amp / max(median_amp, EPS)
        amplitude_rows.append(normalized)
        quality_rows.append(quality)
        if freqs is None:
            freqs = spectrum_freqs
    if freqs is None:
        raise ValueError("无法构造频谱矩阵。")
    return freqs, np.vstack(amplitude_rows), np.asarray(quality_rows, dtype=float)


def compute_single_sided_amplitude(signal: np.ndarray, sampling_rate: float) -> tuple[np.ndarray, np.ndarray]:
    window = np.hanning(len(signal))
    fft_values = np.fft.rfft(signal * window)
    frequencies = np.fft.rfftfreq(len(signal), d=1.0 / sampling_rate)
    amplitudes = (2.0 / np.sum(window)) * np.abs(fft_values)
    if amplitudes.size > 0:
        amplitudes[0] *= 0.5
    return frequencies, amplitudes


def compute_welch_amplitude(
    signal: np.ndarray,
    sampling_rate: float,
    window_seconds: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    if window_seconds is None:
        return compute_single_sided_amplitude(signal, sampling_rate)
    window_size = int(round(window_seconds * sampling_rate))
    if window_size < 32 or len(signal) <= window_size:
        return compute_single_sided_amplitude(signal, sampling_rate)
    step_size = max(window_size // 2, 1)
    power_rows: list[np.ndarray] = []
    freqs: np.ndarray | None = None
    for start in range(0, len(signal) - window_size + 1, step_size):
        sub_freqs, sub_amplitudes = compute_single_sided_amplitude(signal[start : start + window_size], sampling_rate)
        freqs = sub_freqs
        power_rows.append(np.square(sub_amplitudes))
    if not power_rows:
        return compute_single_sided_amplitude(signal, sampling_rate)
    mean_power = np.mean(np.vstack(power_rows), axis=0)
    return freqs, np.sqrt(np.maximum(mean_power, 0.0))


def aggregate_peak_curve(
    search_grid: np.ndarray,
    freqs: np.ndarray,
    amplitudes: np.ndarray,
    *,
    harmonic: int,
) -> np.ndarray:
    curve = np.zeros_like(search_grid)
    for channel_amplitudes in amplitudes:
        curve += np.interp(harmonic * search_grid, freqs, channel_amplitudes, left=0.0, right=0.0)
    return curve / max(amplitudes.shape[0], 1)


def build_harmonic_template_curve(
    *,
    search_grid: np.ndarray,
    freqs: np.ndarray,
    amplitudes: np.ndarray,
    harmonic_weights: tuple[float, ...],
) -> np.ndarray:
    curve = np.zeros_like(search_grid)
    weight_sum = max(float(np.sum(harmonic_weights)), EPS)
    for channel_amplitudes in amplitudes:
        channel_curve = np.zeros_like(search_grid)
        for harmonic_index, weight in enumerate(harmonic_weights, start=1):
            channel_curve += weight * np.interp(
                harmonic_index * search_grid,
                freqs,
                channel_amplitudes,
                left=0.0,
                right=0.0,
            )
        curve += channel_curve / weight_sum
    return curve / max(amplitudes.shape[0], 1)


def build_vote_backprojection_curve(
    *,
    search_grid: np.ndarray,
    freqs: np.ndarray,
    amplitudes: np.ndarray,
    harmonic_weights: tuple[float, ...],
    peak_count: int,
) -> np.ndarray:
    sigma = 0.05
    curve = np.zeros_like(search_grid)
    band_mask = (freqs >= 0.5) & (freqs <= MAX_ANALYSIS_HZ)
    band_freqs = freqs[band_mask]
    for channel_amplitudes in amplitudes:
        band_amps = channel_amplitudes[band_mask]
        if band_amps.size == 0:
            continue
        peak_indices = np.argsort(band_amps)[-peak_count:]
        peak_indices = peak_indices[np.argsort(band_amps[peak_indices])[::-1]]
        peak_amps = band_amps[peak_indices]
        peak_freqs = band_freqs[peak_indices]
        peak_norm = max(float(peak_amps[0]), EPS)
        for peak_freq, peak_amp in zip(peak_freqs, peak_amps):
            amp_weight = float(peak_amp / peak_norm)
            for harmonic_index, harmonic_weight in enumerate(harmonic_weights, start=1):
                candidate_hz = float(peak_freq / harmonic_index)
                if candidate_hz < SEARCH_LOW_HZ or candidate_hz > SEARCH_HIGH_HZ:
                    continue
                curve += amp_weight * harmonic_weight * np.exp(
                    -0.5 * np.square((search_grid - candidate_hz) / sigma)
                )
    return curve / max(amplitudes.shape[0], 1)


def build_autocorr_prior(search_grid: np.ndarray, values: np.ndarray, sampling_rate: float) -> np.ndarray:
    priors: list[np.ndarray] = []
    periods = 1.0 / search_grid
    for channel_values in values:
        centered = channel_values - np.mean(channel_values)
        std = float(np.std(centered))
        if std <= 0:
            continue
        normalized = centered / std
        acf = np.correlate(normalized, normalized, mode="full")[len(normalized) - 1 :]
        if acf.size == 0 or acf[0] <= 0:
            continue
        acf = acf / acf[0]
        lag_seconds = np.arange(acf.size, dtype=float) / sampling_rate
        prior = np.interp(periods, lag_seconds, acf, left=0.0, right=0.0)
        priors.append(normalize_curve(np.maximum(prior, 0.0)))
    if not priors:
        return np.zeros_like(search_grid)
    return np.mean(np.vstack(priors), axis=0)


def dominant_harmonic_from_curve(
    *,
    search_grid: np.ndarray,
    freqs: np.ndarray,
    amplitudes: np.ndarray,
    harmonic_weights: tuple[float, ...],
    score_grid: np.ndarray,
) -> int:
    best_hz = float(search_grid[int(np.argmax(score_grid))])
    mean_amplitude = np.mean(amplitudes, axis=0)
    contributions = []
    for harmonic_index, weight in enumerate(harmonic_weights, start=1):
        contributions.append(
            weight
            * float(
                np.interp(
                    harmonic_index * best_hz,
                    freqs,
                    mean_amplitude,
                    left=0.0,
                    right=0.0,
                )
            )
        )
    return int(np.argmax(contributions) + 1)


def select_top_quality_channels(quality: np.ndarray, top_k: int) -> np.ndarray:
    if quality.size <= top_k:
        return np.arange(quality.size, dtype=int)
    return np.argsort(quality)[-top_k:]


def normalize_curve(values: np.ndarray) -> np.ndarray:
    max_value = float(np.max(values)) if values.size else 0.0
    if max_value <= 0:
        return np.zeros_like(values)
    return values / max_value


def estimate_sampling_rate(frame: pd.DataFrame) -> float:
    time_seconds = (frame["time"] - frame["time"].iloc[0]).dt.total_seconds().to_numpy(dtype=float)
    diffs = np.diff(time_seconds)
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return 50.0
    return float(1.0 / np.median(diffs))


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    values = values[order]
    weights = np.maximum(weights[order], EPS)
    cumulative = np.cumsum(weights)
    cutoff = 0.5 * cumulative[-1]
    index = int(np.searchsorted(cumulative, cutoff, side="left"))
    return float(values[min(index, len(values) - 1)])


def weighted_runner_up(values: np.ndarray, weights: np.ndarray, best_value: float) -> float:
    if values.size == 1:
        return float(best_value)
    mask = np.abs(values - best_value) > 3.0
    if not mask.any():
        alt = np.sort(values)
        return float(alt[-2] if alt[-1] == best_value else alt[-1])
    return weighted_median(values[mask], weights[mask])


def confidence_from_candidates(
    values: np.ndarray,
    weights: np.ndarray,
    best_value: float,
    runner_up_value: float,
) -> float:
    spread = float(np.average(np.abs(values - best_value), weights=np.maximum(weights, EPS)))
    gap = abs(best_value - runner_up_value)
    return float(gap / max(gap + spread, 1.0))


def runner_up_index(score_grid: np.ndarray, best_index: int) -> int:
    if score_grid.size == 1:
        return best_index
    exclusion = max(int(round(0.10 / SEARCH_STEP_HZ)), 1)
    candidate_indices = np.arange(score_grid.size)
    mask = np.abs(candidate_indices - best_index) > exclusion
    if not mask.any():
        sorted_indices = np.argsort(score_grid)
        return int(sorted_indices[-2])
    masked_indices = candidate_indices[mask]
    return int(masked_indices[np.argmax(score_grid[mask])])


def local_confidence(score_grid: np.ndarray, best_index: int) -> float:
    best_score = float(score_grid[best_index])
    if best_score <= 0:
        return 0.0
    second_score = float(score_grid[runner_up_index(score_grid, best_index)])
    return float((best_score - second_score) / max(best_score, EPS))


def dominant_vote(values: list[int]) -> int:
    if not values:
        return 1
    return int(pd.Series(values).value_counts().index[0])


def build_summary(case_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for domain_name, domain_block in list(case_df.groupby("domain", sort=True)) + [("all_labeled", case_df)]:
        for variant_name, block in domain_block.groupby("variant_name", sort=True):
            errors = block["signed_error"].to_numpy(dtype=float)
            rows.append(
                {
                    "domain": domain_name,
                    "variant_name": variant_name,
                    "case_mae": float(np.mean(np.abs(errors))),
                    "case_rmse": float(np.sqrt(np.mean(np.square(errors)))),
                    "mean_signed_error": float(np.mean(errors)),
                    "confidence_mean": float(block["confidence"].mean()),
                    "case_count": int(len(block)),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["domain", "case_mae", "case_rmse", "variant_name"]
    ).reset_index(drop=True)


def build_best_variant_by_domain(summary_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.Series] = []
    for _, block in summary_df.groupby("domain", sort=True):
        rows.append(block.sort_values(["case_mae", "case_rmse", "variant_name"]).iloc[0])
    return pd.DataFrame(rows).reset_index(drop=True)


def build_failure_cases(case_df: pd.DataFrame, best_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for _, best_row in best_df.iterrows():
        block = case_df.loc[
            (case_df["domain"] == best_row["domain"])
            & (case_df["variant_name"] == best_row["variant_name"])
        ].copy()
        block.insert(0, "best_variant_for_domain", best_row["variant_name"])
        rows.append(
            block.sort_values(["abs_error", "case_id"], ascending=[False, True]).head(5)
        )
    return pd.concat(rows, ignore_index=True)


def write_summary_markdown(
    output_path: Path,
    summary_df: pd.DataFrame,
    best_df: pd.DataFrame,
    failure_df: pd.DataFrame,
) -> None:
    lines = [f"# {TRY_NAME} 结论", ""]
    for _, row in best_df.iterrows():
        lines.extend(
            [
                f"## {row['domain']} 最优",
                "",
                f"- 变体：`{row['variant_name']}`",
                f"- case_mae：`{row['case_mae']:.4f}`",
                f"- case_rmse：`{row['case_rmse']:.4f}`",
                f"- mean_signed_error：`{row['mean_signed_error']:.4f}`",
                f"- confidence_mean：`{row['confidence_mean']:.4f}`",
                "",
            ]
        )
    lines.extend(["## 全量排序（按 case_mae）", ""])
    for domain, block in summary_df.groupby("domain", sort=True):
        lines.append(f"### {domain}")
        lines.append("")
        for _, row in block.iterrows():
            lines.append(
                f"- `{row['variant_name']}`: case_mae=`{row['case_mae']:.4f}`, "
                f"case_rmse=`{row['case_rmse']:.4f}`, confidence_mean=`{row['confidence_mean']:.4f}`"
            )
        lines.append("")
    lines.extend(["## 各域最优变体的 hardest cases", ""])
    for domain, block in failure_df.groupby("domain", sort=True):
        lines.append(f"### {domain}")
        lines.append("")
        for _, row in block.iterrows():
            lines.append(
                f"- `case {int(row['case_id'])}`: pred=`{row['pred_rpm']:.2f}`, "
                f"true=`{row['true_rpm']:.2f}`, abs_error=`{row['abs_error']:.2f}`, "
                f"dominant_harmonic=`{int(row['dominant_harmonic'])}`, confidence=`{row['confidence']:.3f}`"
            )
        lines.append("")
    output_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
