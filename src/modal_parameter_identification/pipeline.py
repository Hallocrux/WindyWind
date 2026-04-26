from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.current.data_loading import DatasetRecord
from .fe import compare_modes_with_fe, load_fe_reference
from .io_utils import (
    get_acc_axis_columns,
    get_sensor_columns,
    iter_modal_windows,
    load_case_frame,
    load_case_records,
    load_sync_rpm_series,
    parse_case_id_list,
    resolve_window_rpm,
)
from .models import ModalConfig
from .spectral import (
    align_mode_shape_series,
    build_harmonic_mask,
    compute_fdd_spectrum,
    compute_spectral_matrices,
    estimate_efdd_damping_ratio,
    normalize_mode_shape,
    safe_log10,
    select_peak_index,
    summarize_peak_coherence,
)
from .ssi import assign_mode_clusters, label_stable_poles, run_ssi_cov, select_dominant_cluster

DEFAULT_OUTPUT_DIR = Path("outputs/modal_parameter_identification")


def run_modal_identification(
    *,
    case_ids: list[int] | None = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    sensor_basis: str = "both",
    rpm_source: str = "manifest",
    rpm_series_path: Path | None = None,
    fe_reference_path: Path | None = None,
    config: ModalConfig | None = None,
) -> dict[str, pd.DataFrame]:
    config = config or ModalConfig()
    case_ids = parse_case_id_list(case_ids)
    records, common_columns = load_case_records(case_ids)
    basis_columns = get_sensor_columns(common_columns, sensor_basis=sensor_basis)
    acc_axis_columns = get_acc_axis_columns(common_columns)
    sync_rpm_df = load_sync_rpm_series(rpm_series_path)
    fe_modes = load_fe_reference(fe_reference_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    case_rows: list[dict[str, object]] = []
    window_rows: list[dict[str, object]] = []
    harmonic_rows: list[dict[str, object]] = []
    pole_rows: list[dict[str, object]] = []
    stability_rows: list[dict[str, object]] = []
    strain_shape_rows: list[dict[str, object]] = []
    accy_shape_rows: list[dict[str, object]] = []
    fe_rows: list[dict[str, object]] = []

    for record in records:
        frame = load_case_frame(record, common_columns)
        full_case_diagnostics: dict[str, dict[str, object]] = {}
        basis_summaries: dict[str, dict[str, object]] = {}

        for basis_name, selected_columns in basis_columns.items():
            windows = iter_modal_windows(
                frame,
                selected_columns=selected_columns,
                window_size=config.window_size,
                step_size=config.step_size,
                case_id=record.case_id,
            )
            if not windows:
                basis_summaries[basis_name] = _empty_basis_summary(record.case_id, basis_name)
                continue

            case_matrix = frame[selected_columns].to_numpy(dtype=float)
            case_rpm = record.rpm if rpm_source == "manifest" else resolve_window_rpm(
                record=record,
                start_time=frame["time"].iloc[0],
                end_time=frame["time"].iloc[-1],
                rpm_source=rpm_source,
                sync_rpm_df=sync_rpm_df,
            )
            full_case_diagnostics[basis_name] = _compute_case_spectral_diagnostics(
                matrix=case_matrix,
                sampling_rate=config.sampling_rate,
                rpm=case_rpm,
                config=config,
            )

            basis_window_rows: list[dict[str, object]] = []
            basis_pole_rows: list[dict[str, object]] = []
            selected_shapes: list[np.ndarray] = []
            selected_freqs: list[float] = []
            selected_dampings: list[float] = []

            for window in windows:
                window_rpm = resolve_window_rpm(
                    record=record,
                    start_time=window.start_time,
                    end_time=window.end_time,
                    rpm_source=rpm_source,
                    sync_rpm_df=sync_rpm_df,
                )
                estimate_row, window_harmonic_rows, window_pole_rows = _analyze_window(
                    record=record,
                    basis_name=basis_name,
                    window=window,
                    sampling_rate=config.sampling_rate,
                    rpm=window_rpm,
                    config=config,
                )
                basis_window_rows.append(estimate_row)
                basis_pole_rows.extend(window_pole_rows)
                harmonic_rows.extend(window_harmonic_rows)

                if int(estimate_row["is_valid"]) == 1:
                    selected_shapes.append(np.asarray(estimate_row["mode_shape"], dtype=float))
                    selected_freqs.append(float(estimate_row["selected_frequency_hz"]))
                    if np.isfinite(estimate_row["selected_damping_ratio"]):
                        selected_dampings.append(float(estimate_row["selected_damping_ratio"]))

            basis_summary = _build_basis_summary(
                case_id=record.case_id,
                basis_name=basis_name,
                selected_shapes=selected_shapes,
                selected_freqs=selected_freqs,
                selected_dampings=selected_dampings,
            )
            basis_summaries[basis_name] = basis_summary
            stability_rows.append(_build_stability_row(record.case_id, basis_name, basis_window_rows))

            for pole_row in basis_pole_rows:
                pole_rows.append(pole_row)
            for row in basis_window_rows:
                row = dict(row)
                row.pop("mode_shape", None)
                window_rows.append(row)

            if basis_name == "strain":
                strain_shape_rows.append(_shape_row_from_summary(record.case_id, basis_summary))
            if basis_name == "acc_y":
                accy_shape_rows.append(_shape_row_from_summary(record.case_id, basis_summary))

        _save_case_overview_figure(
            record=record,
            output_dir=output_dir,
            full_case_diagnostics=full_case_diagnostics,
            stability_rows=[row for row in pole_rows if int(row["case_id"]) == record.case_id],
            window_rows=[row for row in window_rows if int(row["case_id"]) == record.case_id],
            basis_summaries=basis_summaries,
            frame=frame,
            acc_axis_columns=acc_axis_columns,
        )

        case_rows.append(_build_case_row(record, rpm_source, basis_summaries))
        for basis_name, summary in basis_summaries.items():
            fe_rows.extend(
                compare_modes_with_fe(
                    case_id=record.case_id,
                    identified_basis=basis_name,
                    identified_frequency_hz=summary.get("frequency_hz"),
                    identified_damping_ratio=summary.get("damping_ratio"),
                    identified_shape=summary.get("mode_shape"),
                    fe_modes=fe_modes,
                )
            )

    case_df = pd.DataFrame(case_rows).sort_values("case_id").reset_index(drop=True)
    window_df = _finalize_frame(window_rows, ["case_id", "basis", "window_index"])
    harmonic_df = _finalize_frame(harmonic_rows, ["case_id", "basis", "window_index", "harmonic_order"])
    pole_df = _finalize_frame(pole_rows, ["case_id", "basis", "window_index", "order", "frequency_hz"])
    stability_df = _finalize_frame(stability_rows, ["case_id", "basis"])
    strain_shape_df = _finalize_frame(strain_shape_rows, ["case_id"])
    accy_shape_df = _finalize_frame(accy_shape_rows, ["case_id"])
    fe_df = pd.DataFrame(fe_rows)

    case_df.to_csv(output_dir / "case_modal_summary.csv", index=False, encoding="utf-8-sig")
    window_df.to_csv(output_dir / "window_modal_estimates.csv", index=False, encoding="utf-8-sig")
    harmonic_df.to_csv(output_dir / "harmonic_mask_table.csv", index=False, encoding="utf-8-sig")
    pole_df.to_csv(output_dir / "stabilization_poles.csv", index=False, encoding="utf-8-sig")
    stability_df.to_csv(output_dir / "stability_statistics.csv", index=False, encoding="utf-8-sig")
    strain_shape_df.to_csv(output_dir / "strain_mode_shapes.csv", index=False, encoding="utf-8-sig")
    accy_shape_df.to_csv(output_dir / "accy_mode_shapes.csv", index=False, encoding="utf-8-sig")
    if not fe_df.empty:
        fe_df.to_csv(output_dir / "fe_comparison.csv", index=False, encoding="utf-8-sig")

    return {
        "case_modal_summary": case_df,
        "window_modal_estimates": window_df,
        "harmonic_mask_table": harmonic_df,
        "stabilization_poles": pole_df,
        "stability_statistics": stability_df,
        "strain_mode_shapes": strain_shape_df,
        "accy_mode_shapes": accy_shape_df,
        "fe_comparison": fe_df,
    }


def _compute_case_spectral_diagnostics(
    *,
    matrix: np.ndarray,
    sampling_rate: float,
    rpm: float | None,
    config: ModalConfig,
) -> dict[str, object]:
    freqs, csd_matrix, coherence_matrix = compute_spectral_matrices(
        matrix,
        sampling_rate=sampling_rate,
        nperseg=min(len(matrix), 1024),
    )
    singular_values, _ = compute_fdd_spectrum(csd_matrix)
    keep_mask, _ = build_harmonic_mask(
        freqs,
        rpm=rpm,
        harmonic_orders=config.harmonic_orders,
        half_width=config.harmonic_half_width,
    )
    masked_curve = singular_values[:, 0].copy()
    masked_curve[~keep_mask] = np.nan
    peak_index = select_peak_index(
        freqs,
        singular_values[:, 0],
        freq_min=config.freq_min,
        freq_max=config.freq_max,
        focus_min=config.focus_min,
        focus_max=config.focus_max,
        keep_mask=keep_mask,
    )
    return {
        "freqs": freqs,
        "singular_curve_raw": singular_values[:, 0],
        "singular_curve_masked": masked_curve,
        "coherence_matrix": coherence_matrix,
        "peak_index": peak_index,
    }


def _analyze_window(
    *,
    record: DatasetRecord,
    basis_name: str,
    window,
    sampling_rate: float,
    rpm: float | None,
    config: ModalConfig,
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    freqs, csd_matrix, coherence_matrix = compute_spectral_matrices(
        window.data,
        sampling_rate=sampling_rate,
        nperseg=min(len(window.data), config.window_size),
    )
    keep_mask, harmonic_band_rows = build_harmonic_mask(
        freqs,
        rpm=rpm,
        harmonic_orders=config.harmonic_orders,
        half_width=config.harmonic_half_width,
    )
    singular_values, singular_vectors = compute_fdd_spectrum(csd_matrix)
    peak_index = select_peak_index(
        freqs,
        singular_values[:, 0],
        freq_min=config.freq_min,
        freq_max=config.freq_max,
        focus_min=config.focus_min,
        focus_max=config.focus_max,
        keep_mask=keep_mask,
    )
    harmonic_rows = [
        {
            "case_id": int(record.case_id),
            "basis": basis_name,
            "window_index": int(window.window_index),
            "segment_id": int(window.segment_id),
            "start_time": window.start_time,
            "end_time": window.end_time,
            "rpm": rpm,
            **row,
        }
        for row in harmonic_band_rows
    ]

    if peak_index is None:
        return _invalid_window_row(record, basis_name, window, rpm), harmonic_rows, []

    fdd_peak_frequency_hz = float(freqs[peak_index])
    efdd_damping_ratio = estimate_efdd_damping_ratio(freqs, singular_values[:, 0], peak_index)
    fdd_shape = normalize_mode_shape(singular_vectors[peak_index, :, 0])

    poles = run_ssi_cov(
        window.data,
        sampling_rate=sampling_rate,
        block_rows=config.ssi_block_rows,
        min_order=config.ssi_min_order,
        max_order=config.ssi_max_order,
        freq_min=config.freq_min,
        freq_max=config.freq_max,
        max_damping_ratio=config.max_damping_ratio,
    )
    poles = label_stable_poles(
        poles,
        freq_tol_hz=config.stabilization_freq_tol_hz,
        damping_tol=config.stabilization_damping_tol,
        mac_tol=config.stabilization_mac_tol,
    )
    poles = assign_mode_clusters(
        poles,
        freq_tol_hz=config.cluster_freq_tol_hz,
        mac_tol=config.stabilization_mac_tol,
    )
    selected_cluster = select_dominant_cluster(
        poles,
        reference_frequency_hz=fdd_peak_frequency_hz,
        focus_min=config.focus_min,
        focus_max=config.focus_max,
    )
    if selected_cluster is not None:
        selected_frequency_hz = float(selected_cluster["frequency_hz"])
        selected_damping_ratio = float(selected_cluster["damping_ratio"])
        selected_shape = np.asarray(selected_cluster["mode_shape"], dtype=float)
        selection_source = "ssi_cov"
    else:
        selected_frequency_hz = fdd_peak_frequency_hz
        selected_damping_ratio = efdd_damping_ratio
        selected_shape = fdd_shape
        selection_source = "fdd_efdd"

    pole_rows = [
        {
            "case_id": int(record.case_id),
            "basis": basis_name,
            "window_index": int(window.window_index),
            "segment_id": int(window.segment_id),
            "start_time": window.start_time,
            "end_time": window.end_time,
            **_serialize_pole_row(pole),
        }
        for pole in poles
    ]

    return (
        {
            "case_id": int(record.case_id),
            "basis": basis_name,
            "window_index": int(window.window_index),
            "segment_id": int(window.segment_id),
            "start_time": window.start_time,
            "end_time": window.end_time,
            "rpm": rpm,
            "fdd_peak_frequency_hz": fdd_peak_frequency_hz,
            "efdd_damping_ratio": efdd_damping_ratio,
            "selected_frequency_hz": selected_frequency_hz,
            "selected_damping_ratio": selected_damping_ratio,
            "selection_source": selection_source,
            "peak_coherence_mean": summarize_peak_coherence(coherence_matrix, peak_index),
            "singular_peak_value": float(singular_values[peak_index, 0]),
            "is_valid": int(np.isfinite(selected_frequency_hz)),
            "mode_shape": selected_shape,
        },
        harmonic_rows,
        pole_rows,
    )


def _invalid_window_row(record: DatasetRecord, basis_name: str, window, rpm: float | None) -> dict[str, object]:
    return {
        "case_id": int(record.case_id),
        "basis": basis_name,
        "window_index": int(window.window_index),
        "segment_id": int(window.segment_id),
        "start_time": window.start_time,
        "end_time": window.end_time,
        "rpm": rpm,
        "fdd_peak_frequency_hz": np.nan,
        "efdd_damping_ratio": np.nan,
        "selected_frequency_hz": np.nan,
        "selected_damping_ratio": np.nan,
        "selection_source": "none",
        "peak_coherence_mean": np.nan,
        "singular_peak_value": np.nan,
        "is_valid": 0,
        "mode_shape": np.zeros(window.data.shape[1], dtype=float),
    }


def _serialize_pole_row(pole: dict[str, object]) -> dict[str, object]:
    row = {key: value for key, value in pole.items() if key != "mode_shape"}
    mode_shape = np.asarray(pole["mode_shape"], dtype=float)
    for index, value in enumerate(mode_shape, start=1):
        row[f"point_{index}"] = float(value)
    return row


def _build_basis_summary(
    *,
    case_id: int,
    basis_name: str,
    selected_shapes: list[np.ndarray],
    selected_freqs: list[float],
    selected_dampings: list[float],
) -> dict[str, object]:
    if not selected_freqs:
        return _empty_basis_summary(case_id, basis_name)
    return {
        "case_id": int(case_id),
        "basis": basis_name,
        "frequency_hz": float(np.median(selected_freqs)),
        "damping_ratio": float(np.median(selected_dampings)) if selected_dampings else np.nan,
        "mode_shape": align_mode_shape_series(selected_shapes),
        "valid_window_count": int(len(selected_freqs)),
        "frequency_iqr_hz": float(np.quantile(selected_freqs, 0.75) - np.quantile(selected_freqs, 0.25))
        if len(selected_freqs) >= 2
        else 0.0,
        "frequency_std_hz": float(np.std(selected_freqs, ddof=0)),
        "damping_iqr": float(np.quantile(selected_dampings, 0.75) - np.quantile(selected_dampings, 0.25))
        if len(selected_dampings) >= 2
        else np.nan,
        "damping_std": float(np.std(selected_dampings, ddof=0)) if selected_dampings else np.nan,
    }


def _empty_basis_summary(case_id: int, basis_name: str) -> dict[str, object]:
    return {
        "case_id": int(case_id),
        "basis": basis_name,
        "frequency_hz": np.nan,
        "damping_ratio": np.nan,
        "mode_shape": np.array([], dtype=float),
        "valid_window_count": 0,
        "frequency_iqr_hz": np.nan,
        "frequency_std_hz": np.nan,
        "damping_iqr": np.nan,
        "damping_std": np.nan,
    }


def _shape_row_from_summary(case_id: int, summary: dict[str, object]) -> dict[str, object]:
    row = {
        "case_id": int(case_id),
        "basis": summary["basis"],
        "frequency_hz": summary["frequency_hz"],
        "damping_ratio": summary["damping_ratio"],
        "valid_window_count": summary["valid_window_count"],
    }
    mode_shape = np.asarray(summary["mode_shape"], dtype=float)
    for index in range(1, 6):
        row[f"point_{index}"] = float(mode_shape[index - 1]) if mode_shape.size >= index else np.nan
    return row


def _build_stability_row(
    case_id: int,
    basis_name: str,
    window_rows: list[dict[str, object]],
) -> dict[str, object]:
    valid_freqs = [float(row["selected_frequency_hz"]) for row in window_rows if int(row["is_valid"]) == 1]
    valid_dampings = [
        float(row["selected_damping_ratio"])
        for row in window_rows
        if int(row["is_valid"]) == 1 and np.isfinite(row["selected_damping_ratio"])
    ]
    return {
        "case_id": int(case_id),
        "basis": basis_name,
        "valid_window_count": len(valid_freqs),
        "frequency_median_hz": float(np.median(valid_freqs)) if valid_freqs else np.nan,
        "frequency_iqr_hz": float(np.quantile(valid_freqs, 0.75) - np.quantile(valid_freqs, 0.25))
        if len(valid_freqs) >= 2
        else np.nan,
        "frequency_std_hz": float(np.std(valid_freqs, ddof=0)) if valid_freqs else np.nan,
        "damping_median": float(np.median(valid_dampings)) if valid_dampings else np.nan,
        "damping_iqr": float(np.quantile(valid_dampings, 0.75) - np.quantile(valid_dampings, 0.25))
        if len(valid_dampings) >= 2
        else np.nan,
        "damping_std": float(np.std(valid_dampings, ddof=0)) if valid_dampings else np.nan,
    }


def _build_case_row(
    record: DatasetRecord,
    rpm_source: str,
    basis_summaries: dict[str, dict[str, object]],
) -> dict[str, object]:
    strain = basis_summaries.get("strain", _empty_basis_summary(record.case_id, "strain"))
    acc_y = basis_summaries.get("acc_y", _empty_basis_summary(record.case_id, "acc_y"))
    return {
        "case_id": int(record.case_id),
        "display_name": record.display_name,
        "rpm_source": rpm_source,
        "rpm": record.rpm,
        "strain_first_frequency_hz": strain["frequency_hz"],
        "strain_damping_ratio": strain["damping_ratio"],
        "strain_valid_window_count": strain["valid_window_count"],
        "accy_first_frequency_hz": acc_y["frequency_hz"],
        "accy_damping_ratio": acc_y["damping_ratio"],
        "accy_valid_window_count": acc_y["valid_window_count"],
        "strain_accy_frequency_gap_hz": (
            abs(float(strain["frequency_hz"]) - float(acc_y["frequency_hz"]))
            if np.isfinite(strain["frequency_hz"]) and np.isfinite(acc_y["frequency_hz"])
            else np.nan
        ),
    }


def _save_case_overview_figure(
    *,
    record: DatasetRecord,
    output_dir: Path,
    full_case_diagnostics: dict[str, dict[str, object]],
    stability_rows: list[dict[str, object]],
    window_rows: list[dict[str, object]],
    basis_summaries: dict[str, dict[str, object]],
    frame: pd.DataFrame,
    acc_axis_columns: dict[str, list[str]],
) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(14, 12), constrained_layout=True)
    basis_names = list(full_case_diagnostics.keys())

    for column_index, basis_name in enumerate(basis_names[:2]):
        diag = full_case_diagnostics[basis_name]
        freqs = np.asarray(diag["freqs"], dtype=float)
        axes[0, column_index].plot(freqs, safe_log10(diag["singular_curve_raw"]), label="raw")
        axes[0, column_index].plot(
            freqs,
            safe_log10(np.nan_to_num(diag["singular_curve_masked"], nan=1e-12)),
            label="masked",
        )
        peak_index = diag.get("peak_index")
        if peak_index is not None:
            axes[0, column_index].axvline(freqs[int(peak_index)], color="tab:red", linestyle="--", alpha=0.7)
        axes[0, column_index].set_title(f"{basis_name} singular spectrum")
        axes[0, column_index].set_xlabel("Frequency (Hz)")
        axes[0, column_index].set_ylabel("log10(S1)")
        axes[0, column_index].grid(True, alpha=0.25)
        axes[0, column_index].legend()

        coherence = np.asarray(diag["coherence_matrix"], dtype=float)
        if peak_index is not None:
            im = axes[1, column_index].imshow(coherence[int(peak_index)], vmin=0.0, vmax=1.0, cmap="viridis")
            axes[1, column_index].set_title(f"{basis_name} coherence @ peak")
            fig.colorbar(im, ax=axes[1, column_index], fraction=0.046, pad=0.04)
        else:
            axes[1, column_index].text(0.5, 0.5, "no peak", ha="center", va="center")
            axes[1, column_index].set_axis_off()

    pole_df = pd.DataFrame(stability_rows)
    if not pole_df.empty:
        for basis_name, group_df in pole_df.groupby("basis"):
            colors = group_df["is_stable"].map({1: "tab:green", 0: "tab:gray"}).tolist()
            axes[2, 0].scatter(group_df["order"], group_df["frequency_hz"], s=14, alpha=0.5, c=colors, label=basis_name)
    axes[2, 0].set_title("Stabilization poles")
    axes[2, 0].set_xlabel("Model order")
    axes[2, 0].set_ylabel("Frequency (Hz)")
    axes[2, 0].grid(True, alpha=0.25)

    valid_window_df = pd.DataFrame(window_rows)
    if not valid_window_df.empty:
        valid_window_df = valid_window_df.loc[valid_window_df["is_valid"] == 1].copy()
    if not valid_window_df.empty:
        labels = sorted(valid_window_df["basis"].unique())
        box_data = [
            valid_window_df.loc[valid_window_df["basis"] == basis_name, "selected_frequency_hz"].to_numpy(dtype=float)
            for basis_name in labels
        ]
        axes[2, 1].boxplot(box_data, tick_labels=labels)
        axes[2, 1].set_ylabel("Frequency (Hz)")
        axes[2, 1].set_title("Window frequency stability")
        axes[2, 1].grid(True, alpha=0.25)
    else:
        axes[2, 1].text(0.5, 0.5, "no valid windows", ha="center", va="center")
        axes[2, 1].set_axis_off()

    fig.suptitle(f"Case {record.case_id} modal overview", fontsize=14)
    fig.savefig(output_dir / f"case_{record.case_id:02d}_modal_overview.png", dpi=180)
    plt.close(fig)
    _save_case_mode_shape_figure(record, output_dir, basis_summaries)


def _save_case_mode_shape_figure(
    record: DatasetRecord,
    output_dir: Path,
    basis_summaries: dict[str, dict[str, object]],
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    has_shape = False
    for basis_name, summary in basis_summaries.items():
        shape = np.asarray(summary.get("mode_shape", []), dtype=float)
        if shape.size == 0:
            continue
        has_shape = True
        ax.plot(np.arange(1, shape.size + 1), shape, marker="o", label=basis_name)
    if has_shape:
        ax.set_title(f"Case {record.case_id} mode shape comparison")
        ax.set_xlabel("Point index")
        ax.set_ylabel("Normalized amplitude")
        ax.grid(True, alpha=0.25)
        ax.legend()
    else:
        ax.text(0.5, 0.5, "no mode shape", ha="center", va="center")
        ax.set_axis_off()
    fig.savefig(output_dir / f"case_{record.case_id:02d}_mode_shape_comparison.png", dpi=180)
    plt.close(fig)


def _finalize_frame(rows: list[dict[str, object]], sort_columns: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values(sort_columns).reset_index(drop=True)
