from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from src.current.data_loading import (
    DatasetRecord,
    get_common_signal_columns,
    load_clean_signal_frame,
    scan_dataset_records,
)
from .models import ModalWindow

TIME_COLUMN = "time"


def load_case_records(case_ids: list[int] | None = None) -> tuple[list[DatasetRecord], list[str]]:
    records = scan_dataset_records()
    if case_ids:
        wanted = set(case_ids)
        records = [record for record in records if record.case_id in wanted]
    if not records:
        raise ValueError("未找到任何待处理工况。")
    common_columns = get_common_signal_columns(records)
    return records, common_columns


def parse_case_id_list(values: list[int] | None) -> list[int] | None:
    if not values:
        return None
    return sorted({int(value) for value in values})


def get_sensor_columns(
    common_columns: list[str],
    *,
    sensor_basis: str,
) -> dict[str, list[str]]:
    strain_columns = sorted(
        [column for column in common_columns if column.startswith("应变传感器")],
        key=_extract_sensor_index,
    )
    acc_y_columns = sorted(
        [column for column in common_columns if column.endswith("AccY")],
        key=_extract_sensor_index,
    )
    if len(strain_columns) != 5:
        raise ValueError(f"应变测点数异常，当前为 {len(strain_columns)}。")
    if len(acc_y_columns) != 5:
        raise ValueError(f"AccY 测点数异常，当前为 {len(acc_y_columns)}。")

    if sensor_basis == "strain":
        return {"strain": strain_columns}
    if sensor_basis == "acc_y":
        return {"acc_y": acc_y_columns}
    if sensor_basis == "both":
        return {"strain": strain_columns, "acc_y": acc_y_columns}
    raise ValueError(f"不支持的 sensor_basis: {sensor_basis}")


def get_acc_axis_columns(common_columns: list[str]) -> dict[str, list[str]]:
    columns_by_axis: dict[str, list[str]] = {}
    for axis in ("X", "Y", "Z"):
        axis_columns = sorted(
            [column for column in common_columns if column.endswith(f"Acc{axis}")],
            key=_extract_sensor_index,
        )
        if axis_columns:
            columns_by_axis[axis] = axis_columns
    return columns_by_axis


def load_case_frame(record: DatasetRecord, common_columns: list[str]) -> pd.DataFrame:
    return load_clean_signal_frame(record, common_columns)


def iter_modal_windows(
    frame: pd.DataFrame,
    *,
    selected_columns: list[str],
    window_size: int,
    step_size: int,
    case_id: int,
) -> list[ModalWindow]:
    windows: list[ModalWindow] = []
    window_index = 0
    for segment_id, segment_df in frame.groupby("__segment_id", sort=True):
        segment_df = segment_df.reset_index(drop=True)
        if len(segment_df) < window_size:
            continue
        for start in range(0, len(segment_df) - window_size + 1, step_size):
            end = start + window_size
            window = segment_df.iloc[start:end].reset_index(drop=True)
            windows.append(
                ModalWindow(
                    case_id=case_id,
                    window_index=window_index,
                    segment_id=int(segment_id),
                    start_time=window[TIME_COLUMN].iloc[0],
                    end_time=window[TIME_COLUMN].iloc[-1],
                    data=window[selected_columns].to_numpy(dtype=float),
                )
            )
            window_index += 1
    return windows


def load_sync_rpm_series(path: Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    rpm_df = pd.read_csv(path)
    required = {"time", "rpm"}
    missing = sorted(required - set(rpm_df.columns))
    if missing:
        raise ValueError(f"同步 rpm 文件缺少必要列: {', '.join(missing)}")
    rpm_df = rpm_df.copy()
    rpm_df["time"] = pd.to_datetime(rpm_df["time"], errors="coerce")
    rpm_df["rpm"] = pd.to_numeric(rpm_df["rpm"], errors="coerce")
    rpm_df = rpm_df.dropna(subset=["time", "rpm"]).reset_index(drop=True)
    if "case_id" in rpm_df.columns:
        rpm_df["case_id"] = pd.to_numeric(rpm_df["case_id"], errors="coerce").astype("Int64")
    return rpm_df


def resolve_window_rpm(
    *,
    record: DatasetRecord,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    rpm_source: str,
    sync_rpm_df: pd.DataFrame | None,
) -> float | None:
    if rpm_source == "manifest":
        return record.rpm
    if rpm_source != "sync_csv":
        raise ValueError(f"不支持的 rpm_source: {rpm_source}")
    if sync_rpm_df is None:
        return None

    rpm_df = sync_rpm_df
    if "case_id" in rpm_df.columns:
        case_df = rpm_df.loc[rpm_df["case_id"] == record.case_id].copy()
        if not case_df.empty:
            rpm_df = case_df
    window_df = rpm_df.loc[(rpm_df["time"] >= start_time) & (rpm_df["time"] <= end_time)].copy()
    if window_df.empty:
        return None
    return float(window_df["rpm"].mean())


def _extract_sensor_index(column_name: str) -> int:
    matches = re.findall(r"(\d+)", column_name)
    if not matches:
        return 9999
    return int(matches[-1])
