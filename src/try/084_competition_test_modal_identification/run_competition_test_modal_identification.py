from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.current.data_loading import DatasetRecord, get_common_signal_columns, prepare_clean_signal_frame, scan_dataset_records
from src.modal_parameter_identification.animation import save_mode_shape_animations
from src.modal_parameter_identification.io_utils import get_acc_axis_columns, get_sensor_columns, iter_modal_windows
from src.modal_parameter_identification.models import ModalConfig
from src.modal_parameter_identification.pipeline import (
    _analyze_window,
    _build_basis_summary,
    _build_case_row,
    _build_stability_row,
    _compute_case_spectral_diagnostics,
    _finalize_frame,
    _save_case_overview_figure,
    _shape_row_from_summary,
)

TRY_NAME = "084_competition_test_modal_identification"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "try" / TRY_NAME
DEFAULT_TEST_PATH = REPO_ROOT / "data" / "test" / "竞赛预测频率工况.csv"
TEST_CASE_ID = 9001


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="对竞赛预测频率工况执行单文件模态识别。")
    parser.add_argument("--test-path", type=Path, default=DEFAULT_TEST_PATH, help="待识别的测试 CSV。")
    parser.add_argument("--rpm", type=float, default=None, help="可选工况级 rpm；未提供时不做谐波屏蔽。")
    parser.add_argument("--freq-min", type=float, default=0.5, help="预分析频率下限。")
    parser.add_argument("--freq-max", type=float, default=6.0, help="预分析频率上限。")
    parser.add_argument("--window-seconds", type=float, default=20.0, help="SSI 窗口长度（秒）。")
    parser.add_argument("--step-seconds", type=float, default=10.0, help="SSI 步长（秒）。")
    parser.add_argument("--harmonic-orders", nargs="+", type=int, default=[1, 2, 3, 4], help="需要标记和屏蔽的谐波阶次。")
    parser.add_argument("--harmonic-half-width", type=float, default=0.2, help="每个谐波屏蔽半宽（Hz）。")
    parser.add_argument("--sensor-basis", choices=("strain", "acc_y", "both"), default="both", help="默认输出的模态测点基底。")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="输出目录。")
    parser.add_argument("--save-mode-shape-animation", action="store_true", help="显式导出振型动画。")
    parser.add_argument("--animation-format", choices=("auto", "mp4", "gif"), default="auto", help="动画写出格式。")
    parser.add_argument("--animation-fps", type=int, default=15, help="动画帧率。")
    parser.add_argument("--animation-cycles", type=int, default=2, help="动画播放周期数。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.test_path.exists():
        raise FileNotFoundError(f"未找到测试文件: {args.test_path}")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    config = ModalConfig(
        freq_min=args.freq_min,
        freq_max=args.freq_max,
        window_seconds=args.window_seconds,
        step_seconds=args.step_seconds,
        harmonic_orders=tuple(args.harmonic_orders),
        harmonic_half_width=args.harmonic_half_width,
    )

    test_record = DatasetRecord(
        case_id=TEST_CASE_ID,
        display_name=args.test_path.stem,
        file_name=args.test_path.name,
        file_path=args.test_path,
        wind_speed=None,
        rpm=None if args.rpm is None else float(args.rpm),
        is_labeled=False,
        original_file_name=args.test_path.name,
        label_source="user_request_2026-04-09",
        notes="competition_test_modal_identification",
    )

    reference_records = scan_dataset_records()
    common_columns = get_common_signal_columns([*reference_records, test_record])
    frame, cleaning_stats = prepare_clean_signal_frame(test_record, common_columns)
    basis_columns = get_sensor_columns(common_columns, sensor_basis=args.sensor_basis)
    acc_axis_columns = get_acc_axis_columns(common_columns)

    case_rows: list[dict[str, object]] = []
    window_rows: list[dict[str, object]] = []
    harmonic_rows: list[dict[str, object]] = []
    pole_rows: list[dict[str, object]] = []
    stability_rows: list[dict[str, object]] = []
    strain_shape_rows: list[dict[str, object]] = []
    accy_shape_rows: list[dict[str, object]] = []

    full_case_diagnostics: dict[str, dict[str, object]] = {}
    basis_summaries: dict[str, dict[str, object]] = {}
    for basis_name, selected_columns in basis_columns.items():
        windows = iter_modal_windows(
            frame,
            selected_columns=selected_columns,
            window_size=config.window_size,
            step_size=config.step_size,
            case_id=test_record.case_id,
        )
        if not windows:
            continue

        case_matrix = frame[selected_columns].to_numpy(dtype=float)
        full_case_diagnostics[basis_name] = _compute_case_spectral_diagnostics(
            matrix=case_matrix,
            sampling_rate=config.sampling_rate,
            rpm=test_record.rpm,
            config=config,
        )

        basis_window_rows: list[dict[str, object]] = []
        basis_pole_rows: list[dict[str, object]] = []
        selected_shapes: list[np.ndarray] = []
        selected_freqs: list[float] = []
        selected_dampings: list[float] = []

        for window in windows:
            estimate_row, window_harmonic_rows, window_pole_rows = _analyze_window(
                record=test_record,
                basis_name=basis_name,
                window=window,
                sampling_rate=config.sampling_rate,
                rpm=test_record.rpm,
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
            case_id=test_record.case_id,
            basis_name=basis_name,
            selected_shapes=selected_shapes,
            selected_freqs=selected_freqs,
            selected_dampings=selected_dampings,
        )
        basis_summaries[basis_name] = basis_summary
        stability_rows.append(_build_stability_row(test_record.case_id, basis_name, basis_window_rows))

        pole_rows.extend(basis_pole_rows)
        for row in basis_window_rows:
            clean_row = dict(row)
            clean_row.pop("mode_shape", None)
            window_rows.append(clean_row)

        if basis_name == "strain":
            strain_shape_rows.append(_shape_row_from_summary(test_record.case_id, basis_summary))
        if basis_name == "acc_y":
            accy_shape_rows.append(_shape_row_from_summary(test_record.case_id, basis_summary))

    _save_case_overview_figure(
        record=test_record,
        output_dir=output_dir,
        full_case_diagnostics=full_case_diagnostics,
        stability_rows=[row for row in pole_rows if int(row["case_id"]) == test_record.case_id],
        window_rows=[row for row in window_rows if int(row["case_id"]) == test_record.case_id],
        basis_summaries=basis_summaries,
        frame=frame,
        acc_axis_columns=acc_axis_columns,
    )

    case_rows.append(_build_case_row(test_record, "manifest", basis_summaries))
    case_df = pd.DataFrame(case_rows).sort_values("case_id").reset_index(drop=True)
    window_df = _finalize_frame(window_rows, ["case_id", "basis", "window_index"])
    harmonic_df = _finalize_frame(harmonic_rows, ["case_id", "basis", "window_index", "harmonic_order"])
    pole_df = _finalize_frame(pole_rows, ["case_id", "basis", "window_index", "order", "frequency_hz"])
    stability_df = _finalize_frame(stability_rows, ["case_id", "basis"])
    strain_shape_df = _finalize_frame(strain_shape_rows, ["case_id"])
    accy_shape_df = _finalize_frame(accy_shape_rows, ["case_id"])

    case_df.to_csv(output_dir / "case_modal_summary.csv", index=False, encoding="utf-8-sig")
    window_df.to_csv(output_dir / "window_modal_estimates.csv", index=False, encoding="utf-8-sig")
    harmonic_df.to_csv(output_dir / "harmonic_mask_table.csv", index=False, encoding="utf-8-sig")
    pole_df.to_csv(output_dir / "stabilization_poles.csv", index=False, encoding="utf-8-sig")
    stability_df.to_csv(output_dir / "stability_statistics.csv", index=False, encoding="utf-8-sig")
    strain_shape_df.to_csv(output_dir / "strain_mode_shapes.csv", index=False, encoding="utf-8-sig")
    accy_shape_df.to_csv(output_dir / "accy_mode_shapes.csv", index=False, encoding="utf-8-sig")

    animation_rows: list[dict[str, object]] = []
    if args.save_mode_shape_animation:
        animation_rows = save_mode_shape_animations(
            output_dir=output_dir,
            shape_tables={
                "strain": strain_shape_df,
                "accy": accy_shape_df,
            },
            animation_format=args.animation_format,
            fps=args.animation_fps,
            cycles=args.animation_cycles,
        )

    write_signal_inventory(
        output_path=output_dir / "signal_inventory.json",
        test_record=test_record,
        frame=frame,
        cleaning_stats=cleaning_stats,
        common_columns=common_columns,
        basis_columns=basis_columns,
        config=config,
    )
    write_summary_markdown(
        output_path=output_dir / "modal_summary.md",
        test_record=test_record,
        config=config,
        case_df=case_df,
        stability_df=stability_df,
        animation_rows=animation_rows,
    )

    print(f"已完成单文件模态识别，输出目录: {output_dir}")
    if not case_df.empty:
        print(
            case_df[
                [
                    "case_id",
                    "strain_first_frequency_hz",
                    "strain_damping_ratio",
                    "accy_first_frequency_hz",
                    "accy_damping_ratio",
                ]
            ].to_string(index=False)
        )


def write_signal_inventory(
    *,
    output_path: Path,
    test_record: DatasetRecord,
    frame: pd.DataFrame,
    cleaning_stats,
    common_columns: list[str],
    basis_columns: dict[str, list[str]],
    config: ModalConfig,
) -> None:
    raw_df = pd.read_csv(test_record.file_path)
    payload = {
        "test_file": str(test_record.file_path),
        "display_name": test_record.display_name,
        "rpm": None if test_record.rpm is None else float(test_record.rpm),
        "raw_row_count": int(len(raw_df)),
        "cleaned_row_count": int(len(frame)),
        "segment_count": int(frame["__segment_id"].nunique()),
        "common_signal_column_count": int(len(common_columns)),
        "common_signal_columns": common_columns,
        "basis_columns": basis_columns,
        "sampling_rate_hz": float(config.sampling_rate),
        "window_seconds": float(config.window_seconds),
        "step_seconds": float(config.step_seconds),
        "cleaning_stats": {
            "leading_missing_len": int(cleaning_stats.leading_missing_len),
            "trailing_missing_len": int(cleaning_stats.trailing_missing_len),
            "edge_removed_rows": int(cleaning_stats.edge_removed_rows),
            "rows_after_edge_drop": int(cleaning_stats.rows_after_edge_drop),
            "internal_short_gap_rows": int(cleaning_stats.internal_short_gap_rows),
            "internal_long_gap_rows_dropped": int(cleaning_stats.internal_long_gap_rows_dropped),
            "continuous_segment_count": int(cleaning_stats.continuous_segment_count),
            "rows_after_long_gap_drop": int(cleaning_stats.rows_after_long_gap_drop),
        },
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_summary_markdown(
    *,
    output_path: Path,
    test_record: DatasetRecord,
    config: ModalConfig,
    case_df: pd.DataFrame,
    stability_df: pd.DataFrame,
    animation_rows: list[dict[str, object]],
) -> None:
    case_row = case_df.iloc[0].to_dict() if not case_df.empty else {}
    lines = [
        "# competition test modal identification",
        "",
        "## 输入口径",
        "",
        f"- 测试文件：`{test_record.file_path}`",
        f"- 工况级 rpm：`{'未提供' if test_record.rpm is None else f'{float(test_record.rpm):.4f}'}`",
        f"- 默认采样率：`{float(config.sampling_rate):.1f} Hz`",
        f"- 预分析频带：`{float(config.freq_min):.1f}-{float(config.freq_max):.1f} Hz`",
        f"- 关注频带：`{float(config.focus_min):.1f}-{float(config.focus_max):.1f} Hz`",
        f"- 窗长 / 步长：`{float(config.window_seconds):.1f}s / {float(config.step_seconds):.1f}s`",
        "",
        "## 工况级结果",
        "",
    ]
    if case_row:
        lines.extend(
            [
                f"- `strain_first_frequency_hz`：`{format_float(case_row.get('strain_first_frequency_hz'))}`",
                f"- `strain_damping_ratio`：`{format_float(case_row.get('strain_damping_ratio'))}`",
                f"- `strain_valid_window_count`：`{int(case_row.get('strain_valid_window_count', 0))}`",
                f"- `accy_first_frequency_hz`：`{format_float(case_row.get('accy_first_frequency_hz'))}`",
                f"- `accy_damping_ratio`：`{format_float(case_row.get('accy_damping_ratio'))}`",
                f"- `accy_valid_window_count`：`{int(case_row.get('accy_valid_window_count', 0))}`",
                f"- `strain_accy_frequency_gap_hz`：`{format_float(case_row.get('strain_accy_frequency_gap_hz'))}`",
            ]
        )
    else:
        lines.append("- 没有得到有效的工况级结果。")

    if not stability_df.empty:
        lines.extend(["", "## 窗口稳定性", ""])
        for _, row in stability_df.sort_values("basis").iterrows():
            lines.append(
                f"- `{row['basis']}`: valid_windows=`{int(row['valid_window_count'])}`, "
                f"freq_median_hz=`{format_float(row['frequency_median_hz'])}`, "
                f"freq_iqr_hz=`{format_float(row['frequency_iqr_hz'])}`, "
                f"damping_median=`{format_float(row['damping_median'])}`"
            )

    if animation_rows:
        lines.extend(["", "## 动画导出", ""])
        for row in animation_rows:
            lines.append(
                f"- `basis={row['basis']}`: status=`{row['status']}`, path=`{row['path'] or '-'}`"
            )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_float(value: object) -> str:
    if value is None:
        return "nan"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(number):
        return "nan"
    return f"{number:.4f}"


if __name__ == "__main__":
    main()
