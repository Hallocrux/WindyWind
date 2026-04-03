from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .data_loading import DatasetRecord, get_common_signal_columns, scan_dataset_records
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
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if records is None:
        records = scan_dataset_records()
    if window_config is None:
        window_config = WindowConfig()
    if quality_config is None:
        quality_config = QualityConfig()

    common_columns = get_common_signal_columns(records)
    case_rows: list[dict[str, object]] = []
    missing_rows: list[dict[str, object]] = []

    for record in records:
        raw = pd.read_csv(record.file_path)
        numeric = raw[common_columns].apply(pd.to_numeric, errors="coerce")
        row_missing = numeric.isna().any(axis=1)
        missing_blocks = _collect_missing_blocks(row_missing.to_numpy())
        top_missing = numeric.isna().sum().sort_values(ascending=False)

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
                "leading_missing_len": missing_blocks[0]["length"] if missing_blocks and missing_blocks[0]["start"] == 0 else 0,
                "trailing_missing_len": missing_blocks[-1]["length"] if missing_blocks and missing_blocks[-1]["end"] == len(raw) - 1 else 0,
                "windows_total": _count_windows(len(raw), window_config),
                "windows_with_missing": _count_windows_with_missing(
                    numeric,
                    window_config,
                    min_missing_ratio=0.0,
                ),
                "windows_with_heavy_missing": _count_windows_with_missing(
                    numeric,
                    window_config,
                    min_missing_ratio=quality_config.heavy_missing_window_ratio,
                ),
                "worst_window_missing_ratio": _worst_window_missing_ratio(numeric, window_config),
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
    max_missing_ratio_row = case_df.sort_values("missing_ratio_in_common_cols", ascending=False).iloc[0]
    max_block_row = case_df.sort_values("max_missing_block_len", ascending=False).iloc[0]
    window_dirty_ratio = float(case_df["windows_with_missing"].sum() / case_df["windows_total"].sum())

    return "\n".join(
        [
            f"工况数: {total_cases}",
            f"平均缺失率(共有通道): {avg_missing_ratio:.4%}",
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


def _count_windows(total_rows: int, config: WindowConfig) -> int:
    if total_rows < config.window_size:
        return 0
    return 1 + (total_rows - config.window_size) // config.step_size


def _count_windows_with_missing(
    numeric: pd.DataFrame,
    config: WindowConfig,
    min_missing_ratio: float,
) -> int:
    count = 0
    total_values = config.window_size * numeric.shape[1]
    for start in range(0, len(numeric) - config.window_size + 1, config.step_size):
        window = numeric.iloc[start : start + config.window_size]
        missing_ratio = float(window.isna().sum().sum() / total_values)
        if missing_ratio > min_missing_ratio:
            count += 1
    return count


def _worst_window_missing_ratio(numeric: pd.DataFrame, config: WindowConfig) -> float:
    worst_ratio = 0.0
    total_values = config.window_size * numeric.shape[1]
    for start in range(0, len(numeric) - config.window_size + 1, config.step_size):
        window = numeric.iloc[start : start + config.window_size]
        missing_ratio = float(window.isna().sum().sum() / total_values)
        worst_ratio = max(worst_ratio, missing_ratio)
    return worst_ratio


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
