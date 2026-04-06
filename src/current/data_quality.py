from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .data_loading import (
    CleaningConfig,
    DatasetRecord,
    get_common_signal_columns,
    prepare_clean_signal_frame,
    scan_dataset_records,
)
from .features import WindowConfig

QUALITY_OUTPUT_DIR = Path("outputs")


@dataclass(frozen=True)
class QualityConfig:
    long_gap_threshold: int = 25
    heavy_missing_window_ratio: float = 0.05
    output_dir: Path = QUALITY_OUTPUT_DIR


def build_data_quality_report(
    records: list[DatasetRecord] | None = None,
    window_config: WindowConfig | None = None,
    quality_config: QualityConfig | None = None,
    cleaning_config: CleaningConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if records is None:
        records = scan_dataset_records()
    if window_config is None:
        window_config = WindowConfig()
    if quality_config is None:
        quality_config = QualityConfig()
    if cleaning_config is None:
        cleaning_config = CleaningConfig()

    common_columns = get_common_signal_columns(records)
    case_rows: list[dict[str, object]] = []
    missing_rows: list[dict[str, object]] = []

    for record in records:
        raw = pd.read_csv(record.file_path)
        numeric = raw[common_columns].apply(pd.to_numeric, errors="coerce")
        row_missing = numeric.isna().any(axis=1)
        missing_blocks = _collect_missing_blocks(row_missing.to_numpy())
        top_missing = numeric.isna().sum().sort_values(ascending=False)
        cleaned_signal, cleaning_stats = prepare_clean_signal_frame(
            record=record,
            common_signal_columns=common_columns,
            cleaning_config=cleaning_config,
        )
        window_stats = _summarize_clean_windows(
            cleaned_signal=cleaned_signal,
            common_signal_columns=common_columns,
            window_config=window_config,
            heavy_missing_window_ratio=quality_config.heavy_missing_window_ratio,
        )

        case_rows.append(
            {
                "case_id": record.case_id,
                "file_name": record.file_name,
                "rows": len(raw),
                "common_signal_columns": len(common_columns),
                "missing_cells_in_common_cols": int(numeric.isna().sum().sum()),
                "missing_ratio_in_common_cols": float(numeric.isna().sum().sum() / numeric.size),
                "rows_with_missing": int(row_missing.sum()),
                "missing_block_count": len(missing_blocks),
                "max_missing_block_len": max((block["length"] for block in missing_blocks), default=0),
                "leading_missing_len": cleaning_stats.leading_missing_len,
                "trailing_missing_len": cleaning_stats.trailing_missing_len,
                "edge_removed_rows": cleaning_stats.edge_removed_rows,
                "edge_removed_ratio": float(cleaning_stats.edge_removed_rows / len(raw)),
                "rows_after_edge_drop": cleaning_stats.rows_after_edge_drop,
                "rows_after_edge_drop_ratio": float(cleaning_stats.rows_after_edge_drop / len(raw)),
                "internal_short_gap_rows": cleaning_stats.internal_short_gap_rows,
                "internal_long_gap_rows_dropped": cleaning_stats.internal_long_gap_rows_dropped,
                "continuous_segment_count": cleaning_stats.continuous_segment_count,
                "rows_after_long_gap_drop": cleaning_stats.rows_after_long_gap_drop,
                "windows_total": window_stats["windows_total"],
                "windows_with_missing": window_stats["windows_with_missing"],
                "windows_with_heavy_missing": window_stats["windows_with_heavy_missing"],
                "worst_window_missing_ratio": window_stats["worst_window_missing_ratio"],
                "long_gap_column_count": _count_long_gap_columns(
                    numeric,
                    threshold=quality_config.long_gap_threshold,
                ),
            }
        )

        for column, missing_count in top_missing.items():
            if missing_count <= 0:
                continue
            missing_rows.append(
                {
                    "case_id": record.case_id,
                    "file_name": record.file_name,
                    "column_name": column,
                    "missing_count": int(missing_count),
                    "missing_ratio": float(missing_count / len(raw)),
                    "max_missing_run": _max_missing_run(numeric[column].isna().to_numpy()),
                }
            )

    case_df = pd.DataFrame(case_rows).sort_values("case_id").reset_index(drop=True)
    missing_df = pd.DataFrame(missing_rows).sort_values(
        ["case_id", "missing_count", "column_name"],
        ascending=[True, False, True],
    ).reset_index(drop=True)
    return case_df, missing_df


def save_data_quality_report(
    case_df: pd.DataFrame,
    missing_df: pd.DataFrame,
    output_dir: Path = QUALITY_OUTPUT_DIR,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    case_df.to_csv(output_dir / "data_quality_summary.csv", index=False, encoding="utf-8-sig")
    missing_df.to_csv(output_dir / "data_quality_missing_columns.csv", index=False, encoding="utf-8-sig")


def format_quality_summary(case_df: pd.DataFrame) -> str:
    total_cases = int(len(case_df))
    avg_missing_ratio = float(case_df["missing_ratio_in_common_cols"].mean())
    avg_edge_removed_ratio = float(case_df["edge_removed_ratio"].mean())
    max_missing_ratio_row = case_df.sort_values("missing_ratio_in_common_cols", ascending=False).iloc[0]
    max_block_row = case_df.sort_values("max_missing_block_len", ascending=False).iloc[0]
    total_windows = int(case_df["windows_total"].sum())
    window_dirty_ratio = 0.0
    if total_windows > 0:
        window_dirty_ratio = float(case_df["windows_with_missing"].sum() / total_windows)

    return "\n".join(
        [
            f"工况数: {total_cases}",
            f"平均缺失率(共有通道): {avg_missing_ratio:.4%}",
            f"平均首尾连续缺失删除比例: {avg_edge_removed_ratio:.4%}",
            f"最高缺失率工况: 工况{int(max_missing_ratio_row['case_id'])} ({max_missing_ratio_row['missing_ratio_in_common_cols']:.4%})",
            f"最长连续缺失段: 工况{int(max_block_row['case_id'])} ({int(max_block_row['max_missing_block_len'])} 点)",
            f"受缺失影响窗口占比: {window_dirty_ratio:.4%}",
        ]
    )


def _collect_missing_blocks(mask: np.ndarray) -> list[dict[str, int]]:
    blocks: list[dict[str, int]] = []
    start: int | None = None
    for index, is_missing in enumerate(mask):
        if is_missing and start is None:
            start = index
        elif not is_missing and start is not None:
            blocks.append({"start": start, "end": index - 1, "length": index - start})
            start = None

    if start is not None:
        blocks.append({"start": start, "end": len(mask) - 1, "length": len(mask) - start})
    return blocks


def _summarize_clean_windows(
    cleaned_signal: pd.DataFrame,
    common_signal_columns: list[str],
    window_config: WindowConfig,
    heavy_missing_window_ratio: float,
) -> dict[str, float | int]:
    windows_total = 0
    windows_with_missing = 0
    windows_with_heavy_missing = 0
    worst_window_missing_ratio = 0.0
    total_values = window_config.window_size * len(common_signal_columns)

    for _, segment_df in cleaned_signal.groupby("__segment_id", sort=True):
        segment_df = segment_df.reset_index(drop=True)
        if len(segment_df) < window_config.window_size:
            continue

        for start in range(0, len(segment_df) - window_config.window_size + 1, window_config.step_size):
            window = segment_df.iloc[start : start + window_config.window_size]
            missing_ratio = float(window["__row_missing_count"].sum() / total_values)
            windows_total += 1
            if missing_ratio > 0.0:
                windows_with_missing += 1
            if missing_ratio > heavy_missing_window_ratio:
                windows_with_heavy_missing += 1
            worst_window_missing_ratio = max(worst_window_missing_ratio, missing_ratio)

    return {
        "windows_total": windows_total,
        "windows_with_missing": windows_with_missing,
        "windows_with_heavy_missing": windows_with_heavy_missing,
        "worst_window_missing_ratio": worst_window_missing_ratio,
    }


def _count_long_gap_columns(numeric: pd.DataFrame, threshold: int) -> int:
    return int(
        sum(
            _max_missing_run(numeric[column].isna().to_numpy()) >= threshold
            for column in numeric.columns
        )
    )


def _max_missing_run(mask: np.ndarray) -> int:
    max_run = 0
    current = 0
    for is_missing in mask:
        if is_missing:
            current += 1
            max_run = max(max_run, current)
        else:
            current = 0
    return max_run
