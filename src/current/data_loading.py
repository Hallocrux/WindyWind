from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_DATA_DIR = Path("data/final/datasets")
DEFAULT_MANIFEST_PATH = Path("data/final/dataset_manifest.csv")
TIME_COLUMN = "time"
QUALITY_COLUMNS = {
    "__row_missing_count",
    "__row_has_missing",
    "__in_leading_missing_block",
    "__in_trailing_missing_block",
    "__segment_id",
}
INVALID_COLUMNS = {
    "WSMS00005.AccX",
    "WSMS00005.AccY",
    "WSMS00005.AccZ",
}
MANIFEST_COLUMNS = {
    "case_id",
    "display_name",
    "wind_speed",
    "rpm",
    "original_file_name",
    "label_source",
    "notes",
}


@dataclass(frozen=True)
class DatasetRecord:
    case_id: int
    display_name: str
    file_name: str
    file_path: Path
    wind_speed: float | None
    rpm: float | None
    is_labeled: bool
    original_file_name: str
    label_source: str
    notes: str

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["file_path"] = str(self.file_path)
        return payload


@dataclass(frozen=True)
class CleaningConfig:
    max_middle_interp_gap_rows: int = 5


@dataclass(frozen=True)
class CleaningStats:
    leading_missing_len: int
    trailing_missing_len: int
    edge_removed_rows: int
    rows_after_edge_drop: int
    internal_short_gap_rows: int
    internal_long_gap_rows_dropped: int
    continuous_segment_count: int
    rows_after_long_gap_drop: int


def scan_dataset_records(
    data_dir: Path = DEFAULT_DATA_DIR,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
) -> list[DatasetRecord]:
    manifest_df = _read_manifest(manifest_path)
    records = [_build_record_from_manifest_row(row, data_dir) for _, row in manifest_df.iterrows()]
    if not records:
        raise FileNotFoundError(f"未在 {manifest_path} 找到任何工况记录。")

    existing_files = {path.name for path in data_dir.glob("*.csv")}
    expected_files = {record.file_name for record in records}
    missing_files = sorted(expected_files - existing_files)
    unexpected_files = sorted(existing_files - expected_files)
    if missing_files:
        raise FileNotFoundError(
            f"manifest 中声明的标准数据文件不存在: {', '.join(missing_files)}"
        )
    if unexpected_files:
        raise ValueError(
            f"datasets 目录存在未登记的 CSV 文件: {', '.join(unexpected_files)}"
        )

    return records


def get_common_signal_columns(records: list[DatasetRecord]) -> list[str]:
    headers = [list(pd.read_csv(record.file_path, nrows=0).columns) for record in records]
    common = set(headers[0]) - INVALID_COLUMNS - {TIME_COLUMN}
    for columns in headers[1:]:
        common &= set(columns) - INVALID_COLUMNS - {TIME_COLUMN}

    ordered_common = [column for column in headers[0] if column in common]
    if not ordered_common:
        raise ValueError("未找到所有工况共有的有效传感器列。")
    return ordered_common


def load_clean_signal_frame(
    record: DatasetRecord,
    common_signal_columns: list[str],
    cleaning_config: CleaningConfig | None = None,
) -> pd.DataFrame:
    cleaned, _ = prepare_clean_signal_frame(
        record=record,
        common_signal_columns=common_signal_columns,
        cleaning_config=cleaning_config,
    )
    return cleaned


def prepare_clean_signal_frame(
    record: DatasetRecord,
    common_signal_columns: list[str],
    cleaning_config: CleaningConfig | None = None,
) -> tuple[pd.DataFrame, CleaningStats]:
    if cleaning_config is None:
        cleaning_config = CleaningConfig()

    frame = pd.read_csv(record.file_path)
    if TIME_COLUMN not in frame.columns:
        raise ValueError(f"{record.file_name} 缺少 time 列。")

    cleaned_time = _clean_time_series(frame[TIME_COLUMN])
    cleaned = frame.loc[:, [TIME_COLUMN, *common_signal_columns]].copy()
    cleaned[TIME_COLUMN] = pd.to_datetime(cleaned_time, errors="coerce")
    cleaned = cleaned.dropna(subset=[TIME_COLUMN])
    cleaned = cleaned.sort_values(TIME_COLUMN).drop_duplicates(
        subset=TIME_COLUMN, keep="first"
    )
    cleaned = cleaned.reset_index(drop=True)

    numeric = cleaned[common_signal_columns].apply(pd.to_numeric, errors="coerce")
    row_missing_count = numeric.isna().sum(axis=1)
    row_has_missing = row_missing_count.gt(0)
    leading_mask, trailing_mask = _edge_missing_masks(row_has_missing)
    missing_blocks = _collect_missing_blocks(row_has_missing.to_numpy(dtype=bool, copy=False))

    leading_missing_len = (
        missing_blocks[0]["length"]
        if missing_blocks and missing_blocks[0]["start"] == 0
        else 0
    )
    trailing_missing_len = (
        missing_blocks[-1]["length"]
        if missing_blocks and missing_blocks[-1]["end"] == len(cleaned) - 1
        else 0
    )
    edge_removed_rows = leading_missing_len + trailing_missing_len

    keep_edge_mask = ~(leading_mask.astype(bool) | trailing_mask.astype(bool))
    cleaned = cleaned.loc[keep_edge_mask].reset_index(drop=True)
    numeric = numeric.loc[keep_edge_mask].reset_index(drop=True)

    if cleaned.empty:
        raise ValueError(f"{record.file_name} 清洗后无有效数据。")

    rows_after_edge_drop = len(cleaned)
    trimmed_missing_count = numeric.isna().sum(axis=1)
    trimmed_row_has_missing = trimmed_missing_count.gt(0)
    trimmed_blocks = _collect_missing_blocks(
        trimmed_row_has_missing.to_numpy(dtype=bool, copy=False)
    )

    long_gap_mask = np.zeros(len(cleaned), dtype=bool)
    internal_short_gap_rows = 0
    internal_long_gap_rows_dropped = 0
    for block in trimmed_blocks:
        if block["length"] <= cleaning_config.max_middle_interp_gap_rows:
            internal_short_gap_rows += block["length"]
            continue
        long_gap_mask[block["start"] : block["end"] + 1] = True
        internal_long_gap_rows_dropped += block["length"]

    keep_middle_mask = ~long_gap_mask
    source_indices = np.flatnonzero(keep_middle_mask)
    cleaned = cleaned.loc[keep_middle_mask].reset_index(drop=True)
    numeric = numeric.loc[keep_middle_mask].reset_index(drop=True)

    if cleaned.empty:
        raise ValueError(f"{record.file_name} 清洗后无有效数据。")

    segment_ids = _build_segment_ids(source_indices)
    retained_missing_count = numeric.isna().sum(axis=1)
    retained_row_has_missing = retained_missing_count.gt(0)

    filled_numeric = numeric.copy()
    for segment_id in np.unique(segment_ids):
        segment_mask = segment_ids == segment_id
        segment_numeric = filled_numeric.loc[segment_mask].copy()
        segment_numeric = segment_numeric.interpolate(method="linear", axis=0)
        segment_numeric = segment_numeric.ffill().bfill().fillna(0.0)
        filled_numeric.loc[segment_mask] = segment_numeric.to_numpy()

    cleaned.loc[:, common_signal_columns] = filled_numeric
    cleaned["__row_missing_count"] = retained_missing_count.to_numpy(dtype=int)
    cleaned["__row_has_missing"] = retained_row_has_missing.to_numpy(dtype=int)
    cleaned["__in_leading_missing_block"] = np.zeros(len(cleaned), dtype=int)
    cleaned["__in_trailing_missing_block"] = np.zeros(len(cleaned), dtype=int)
    cleaned["__segment_id"] = segment_ids

    stats = CleaningStats(
        leading_missing_len=leading_missing_len,
        trailing_missing_len=trailing_missing_len,
        edge_removed_rows=edge_removed_rows,
        rows_after_edge_drop=rows_after_edge_drop,
        internal_short_gap_rows=internal_short_gap_rows,
        internal_long_gap_rows_dropped=internal_long_gap_rows_dropped,
        continuous_segment_count=int(segment_ids[-1] + 1) if len(segment_ids) > 0 else 0,
        rows_after_long_gap_drop=len(cleaned),
    )
    return cleaned, stats


def build_metadata_frame(records: list[DatasetRecord]) -> pd.DataFrame:
    return pd.DataFrame([record.to_dict() for record in records]).sort_values(
        ["case_id", "file_name"]
    )


def build_dataset_inventory(records: list[DatasetRecord]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for record in records:
        frame = pd.read_csv(record.file_path)
        time_stats = _summarize_time_column(frame[TIME_COLUMN]) if TIME_COLUMN in frame.columns else {}
        rows.append(
            {
                "case_id": record.case_id,
                "file_name": record.file_name,
                "row_count": int(len(frame)),
                "column_count": int(len(frame.columns)),
                "start_time": time_stats.get("start_time"),
                "end_time": time_stats.get("end_time"),
                "duration_seconds": time_stats.get("duration_seconds"),
                "sampling_hz_est": time_stats.get("sampling_hz_est"),
                "has_invalid_wsms00005": int(any(column in INVALID_COLUMNS for column in frame.columns)),
            }
        )
    return pd.DataFrame(rows).sort_values("case_id").reset_index(drop=True)


def standard_case_file_name(case_id: int) -> str:
    return f"工况{case_id}.csv"


def _clean_time_series(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.strip()
    text = text.str.removeprefix('="').str.removeprefix("=")
    return text.str.strip('"')


def _edge_missing_masks(row_has_missing: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    values = row_has_missing.to_numpy(dtype=bool, copy=False)
    leading = np.zeros(len(values), dtype=int)
    trailing = np.zeros(len(values), dtype=int)

    index = 0
    while index < len(values) and values[index]:
        leading[index] = 1
        index += 1

    index = len(values) - 1
    while index >= 0 and values[index]:
        trailing[index] = 1
        index -= 1

    return leading, trailing


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


def _build_segment_ids(source_indices: np.ndarray) -> np.ndarray:
    if len(source_indices) == 0:
        return np.array([], dtype=int)

    segment_ids = np.zeros(len(source_indices), dtype=int)
    if len(source_indices) == 1:
        return segment_ids

    segment_ids[1:] = np.cumsum(np.diff(source_indices) > 1)
    return segment_ids


def _read_manifest(manifest_path: Path) -> pd.DataFrame:
    if not manifest_path.exists():
        raise FileNotFoundError(f"未找到数据 manifest: {manifest_path}")

    manifest_df = pd.read_csv(manifest_path, dtype=str, keep_default_na=False).fillna("")
    missing_columns = sorted(MANIFEST_COLUMNS - set(manifest_df.columns))
    if missing_columns:
        raise ValueError(
            f"manifest 缺少必要列: {', '.join(missing_columns)}"
        )
    if manifest_df.empty:
        raise ValueError("manifest 为空。")

    duplicated_case_ids = (
        manifest_df["case_id"].astype(str).str.strip().loc[
            lambda series: series.duplicated(keep=False)
        ].tolist()
    )
    if duplicated_case_ids:
        duplicated_text = ", ".join(sorted(set(duplicated_case_ids), key=int))
        raise ValueError(f"manifest 中存在重复的 case_id: {duplicated_text}")

    manifest_df = manifest_df.copy()
    manifest_df["case_id"] = manifest_df["case_id"].map(_parse_case_id)
    manifest_df = manifest_df.sort_values("case_id").reset_index(drop=True)
    return manifest_df


def _build_record_from_manifest_row(row: pd.Series, data_dir: Path) -> DatasetRecord:
    case_id = int(row["case_id"])
    display_name = str(row["display_name"]).strip()
    original_file_name = str(row["original_file_name"]).strip()
    label_source = str(row["label_source"]).strip()
    notes = str(row["notes"]).strip()
    if not display_name:
        raise ValueError(f"case_id={case_id} 缺少 display_name。")
    if not original_file_name:
        raise ValueError(f"case_id={case_id} 缺少 original_file_name。")
    if not label_source:
        raise ValueError(f"case_id={case_id} 缺少 label_source。")

    wind_speed = _parse_optional_float(row["wind_speed"], column_name="wind_speed", case_id=case_id)
    rpm = _parse_optional_float(row["rpm"], column_name="rpm", case_id=case_id)
    file_name = standard_case_file_name(case_id)
    return DatasetRecord(
        case_id=case_id,
        display_name=display_name,
        file_name=file_name,
        file_path=data_dir / file_name,
        wind_speed=wind_speed,
        rpm=rpm,
        is_labeled=wind_speed is not None and rpm is not None,
        original_file_name=original_file_name,
        label_source=label_source,
        notes=notes,
    )


def _parse_case_id(value: object) -> int:
    text = str(value).strip()
    if not text:
        raise ValueError("manifest 存在空的 case_id。")
    try:
        case_id = int(text)
    except ValueError as exc:
        raise ValueError(f"manifest 中存在无法解析的 case_id: {text}") from exc
    if case_id <= 0:
        raise ValueError(f"case_id 必须为正整数，当前为: {case_id}")
    return case_id


def _parse_optional_float(
    value: object,
    *,
    column_name: str,
    case_id: int,
) -> float | None:
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError as exc:
        raise ValueError(
            f"case_id={case_id} 的 {column_name} 不是合法数字: {text}"
        ) from exc


def _summarize_time_column(series: pd.Series) -> dict[str, object]:
    cleaned_time = _clean_time_series(series)
    parsed = pd.to_datetime(cleaned_time, errors="coerce")
    parsed = parsed.dropna().sort_values().drop_duplicates().reset_index(drop=True)
    if parsed.empty:
        return {
            "start_time": pd.NaT,
            "end_time": pd.NaT,
            "duration_seconds": np.nan,
            "sampling_hz_est": np.nan,
        }

    duration_seconds = 0.0
    sampling_hz_est = np.nan
    if len(parsed) >= 2:
        diffs = parsed.diff().dt.total_seconds().dropna()
        positive_diffs = diffs[diffs > 0]
        if not positive_diffs.empty:
            interval = float(positive_diffs.mode().iloc[0])
            if interval > 0:
                sampling_hz_est = 1.0 / interval
        duration_seconds = float((parsed.iloc[-1] - parsed.iloc[0]).total_seconds())

    return {
        "start_time": parsed.iloc[0],
        "end_time": parsed.iloc[-1],
        "duration_seconds": duration_seconds,
        "sampling_hz_est": sampling_hz_est,
    }
