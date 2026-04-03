from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import re

import numpy as np
import pandas as pd

DEFAULT_DATA_DIR = Path("data/final/datasets")
TIME_COLUMN = "time"
QUALITY_COLUMNS = {
    "__row_missing_count",
    "__row_has_missing",
    "__in_leading_missing_block",
    "__in_trailing_missing_block",
}
INVALID_COLUMNS = {
    "WSMS00005.AccX",
    "WSMS00005.AccY",
    "WSMS00005.AccZ",
}
CASE_ID_RE = re.compile(r"工况(?P<id>\d+)")
LABEL_RE = re.compile(
    r"工况(?P<id>\d+).*?风速(?P<ws>[0-9.]+)ms[,，]转速(?P<rpm>[0-9.]+)rpm"
)


@dataclass(frozen=True)
class DatasetRecord:
    case_id: int
    file_name: str
    file_path: Path
    wind_speed: float | None
    rpm: float | None
    is_labeled: bool

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["file_path"] = str(self.file_path)
        return payload


def parse_dataset_record(path: Path) -> DatasetRecord:
    case_match = CASE_ID_RE.search(path.stem)
    if case_match is None:
        raise ValueError(f"无法从文件名解析工况编号: {path.name}")

    label_match = LABEL_RE.search(path.stem)
    if label_match is None:
        return DatasetRecord(
            case_id=int(case_match.group("id")),
            file_name=path.name,
            file_path=path,
            wind_speed=None,
            rpm=None,
            is_labeled=False,
        )

    return DatasetRecord(
        case_id=int(label_match.group("id")),
        file_name=path.name,
        file_path=path,
        wind_speed=float(label_match.group("ws")),
        rpm=float(label_match.group("rpm")),
        is_labeled=True,
    )


def scan_dataset_records(data_dir: Path = DEFAULT_DATA_DIR) -> list[DatasetRecord]:
    records = [parse_dataset_record(path) for path in sorted(data_dir.glob("*.csv"))]
    if not records:
        raise FileNotFoundError(f"未在 {data_dir} 找到 CSV 数据文件。")
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
) -> pd.DataFrame:
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

    numeric = cleaned[common_signal_columns].apply(pd.to_numeric, errors="coerce")
    row_missing_count = numeric.isna().sum(axis=1)
    row_has_missing = row_missing_count.gt(0)
    leading_mask, trailing_mask = _edge_missing_masks(row_has_missing)
    numeric = numeric.interpolate(method="linear", axis=0).ffill().bfill()
    numeric = numeric.fillna(0.0)

    cleaned.loc[:, common_signal_columns] = numeric
    cleaned["__row_missing_count"] = row_missing_count.to_numpy()
    cleaned["__row_has_missing"] = row_has_missing.to_numpy(dtype=int)
    cleaned["__in_leading_missing_block"] = leading_mask
    cleaned["__in_trailing_missing_block"] = trailing_mask
    if cleaned.empty:
        raise ValueError(f"{record.file_name} 清洗后无有效数据。")
    return cleaned.reset_index(drop=True)


def build_metadata_frame(records: list[DatasetRecord]) -> pd.DataFrame:
    return pd.DataFrame([record.to_dict() for record in records]).sort_values(
        ["case_id", "file_name"]
    )


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
