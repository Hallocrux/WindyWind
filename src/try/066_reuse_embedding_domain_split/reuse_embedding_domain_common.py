from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.current.data_loading import (
    DatasetRecord,
    get_common_signal_columns,
    load_clean_signal_frame,
    scan_dataset_records,
)
TRY053_SCRIPT_PATH = REPO_ROOT / "src" / "try" / "053_support_window_residual_quickcheck" / "run_support_window_residual_quickcheck.py"
TRY057_OUTPUT_DIR = REPO_ROOT / "outputs" / "try" / "057_embedding_space_diagnosis"
TRY057_CKPT_DIR = TRY057_OUTPUT_DIR / "models" / "checkpoints"
TRY057_CASE_TABLE_PATH = TRY057_OUTPUT_DIR / "embedding_case_table.csv"
ADDED_MANIFEST_PATH = REPO_ROOT / "data" / "added" / "dataset_manifest.csv"
ADDED_DATA_DIR = REPO_ROOT / "data" / "added" / "standardized_datasets"
ADDED2_MANIFEST_PATH = REPO_ROOT / "data" / "added2" / "dataset_manifest.csv"
ADDED2_DATA_DIR = REPO_ROOT / "data" / "added2" / "standardized_datasets"
WINDOW_LABELS = ("2s", "8s")


@dataclass(frozen=True)
class SourceCatalog:
    final_records: list[DatasetRecord]
    added_records: list[DatasetRecord]
    added2_records: list[DatasetRecord]

    @property
    def all_records(self) -> list[DatasetRecord]:
        return sorted([*self.final_records, *self.added_records, *self.added2_records], key=lambda record: record.case_id)

    @property
    def labeled_records(self) -> list[DatasetRecord]:
        return [record for record in self.all_records if record.is_labeled]


def load_try053_module():
    spec = importlib.util.spec_from_file_location("try053_reuse_embedding_domain_common", TRY053_SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 053 脚本: {TRY053_SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_records_from_manifest(manifest_path: Path, data_dir: Path) -> list[DatasetRecord]:
    manifest_df = pd.read_csv(manifest_path)
    records: list[DatasetRecord] = []
    for _, row in manifest_df.iterrows():
        case_id = int(row["case_id"])
        wind_speed = float(row["wind_speed"]) if not pd.isna(row["wind_speed"]) else None
        rpm = float(row["rpm"]) if not pd.isna(row["rpm"]) else None
        records.append(
            DatasetRecord(
                case_id=case_id,
                display_name=str(row["display_name"]),
                file_name=f"工况{case_id}.csv",
                file_path=data_dir / f"工况{case_id}.csv",
                wind_speed=wind_speed,
                rpm=rpm,
                is_labeled=wind_speed is not None and rpm is not None,
                original_file_name=str(row["original_file_name"]),
                label_source=str(row["label_source"]),
                notes=str(row["notes"]),
            )
        )
    return sorted(records, key=lambda record: record.case_id)


def load_source_catalog() -> SourceCatalog:
    final_records = sorted(scan_dataset_records(), key=lambda record: record.case_id)
    added_records = load_records_from_manifest(ADDED_MANIFEST_PATH, ADDED_DATA_DIR)
    added2_records = load_records_from_manifest(ADDED2_MANIFEST_PATH, ADDED2_DATA_DIR)
    return SourceCatalog(
        final_records=final_records,
        added_records=added_records,
        added2_records=added2_records,
    )


def build_raw_source_domain_map(catalog: SourceCatalog) -> dict[int, str]:
    mapping: dict[int, str] = {}
    for record in catalog.final_records:
        mapping[record.case_id] = "final"
    for record in catalog.added_records:
        mapping[record.case_id] = "added"
    for record in catalog.added2_records:
        mapping[record.case_id] = "added2"
    return mapping


def build_record_table(catalog: SourceCatalog) -> pd.DataFrame:
    raw_source_domain_map = build_raw_source_domain_map(catalog)
    rows = []
    for record in catalog.all_records:
        rows.append(
            {
                "case_id": int(record.case_id),
                "file_name": str(record.file_name),
                "display_name": str(record.display_name),
                "raw_source_domain": raw_source_domain_map[int(record.case_id)],
                "wind_speed": np.nan if record.wind_speed is None else float(record.wind_speed),
                "rpm": np.nan if record.rpm is None else float(record.rpm),
                "is_labeled": bool(record.is_labeled),
                "notes": str(record.notes),
            }
        )
    return pd.DataFrame(rows).sort_values("case_id").reset_index(drop=True)


def build_cleaned_signal_frames(records: list[DatasetRecord]) -> tuple[list[str], dict[int, pd.DataFrame]]:
    common_signal_columns = get_common_signal_columns(records)
    cleaned_signal_frames = {
        record.case_id: load_clean_signal_frame(record, common_signal_columns)
        for record in records
    }
    return common_signal_columns, cleaned_signal_frames


def load_fixed_window_embeddings(
    *,
    try053,
    export_records: list[DatasetRecord],
    cleaned_signal_frames: dict[int, pd.DataFrame],
    checkpoint_root: Path = TRY057_CKPT_DIR,
) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for window_label in WINDOW_LABELS:
        window_config = try053.WINDOW_CONFIGS[window_label]
        export_dataset = try053.build_raw_window_dataset(
            export_records,
            {record.case_id: cleaned_signal_frames[record.case_id] for record in export_records},
            window_config,
        )
        model = try053.TinyTCNEncoderRegressor(in_channels=export_dataset.windows.shape[1])
        ckpt_path = checkpoint_root / f"unified_all_{window_label}.pt"
        norm_path = checkpoint_root / f"unified_all_{window_label}_norm.npz"
        meta_path = checkpoint_root / f"unified_all_{window_label}.json"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"缺少旧 embedding checkpoint: {ckpt_path}")
        if not norm_path.exists():
            raise FileNotFoundError(f"缺少旧 embedding norm: {norm_path}")
        if not meta_path.exists():
            raise FileNotFoundError(f"缺少旧 embedding meta: {meta_path}")

        state = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state)
        norm = np.load(norm_path)
        mean = norm["mean"]
        std = norm["std"]

        x_export = export_dataset.windows.astype(np.float32)
        x_export_norm = ((x_export - mean) / std).astype(np.float32)
        with torch.no_grad():
            export_tensor = torch.from_numpy(x_export_norm).float()
            window_embedding = model.encode(export_tensor).cpu().numpy()

        result[window_label] = {
            "export_meta_df": export_dataset.meta_df.copy().reset_index(drop=True),
            "window_embedding": window_embedding,
            "mean": mean,
            "std": std,
            "meta": json.loads(meta_path.read_text(encoding="utf-8")),
        }
    return result


def aggregate_case_embeddings(meta_df: pd.DataFrame, window_embeddings: np.ndarray, column_name: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for case_id, block in meta_df.groupby("case_id", sort=False):
        indices = block.index.to_numpy(dtype=int, copy=False)
        rows.append(
            {
                "case_id": int(case_id),
                column_name: window_embeddings[indices].mean(axis=0).astype(float),
            }
        )
    return pd.DataFrame(rows).sort_values("case_id").reset_index(drop=True)


def build_embedding_case_table(
    record_df: pd.DataFrame,
    fixed_window_embeddings: dict[str, dict[str, object]],
) -> pd.DataFrame:
    result = record_df.copy()
    for window_label in WINDOW_LABELS:
        embedding_df = aggregate_case_embeddings(
            fixed_window_embeddings[window_label]["export_meta_df"],
            fixed_window_embeddings[window_label]["window_embedding"],
            f"embedding_{window_label}",
        )
        result = result.merge(embedding_df, on="case_id", how="left")

    result["embedding_concat"] = result.apply(
        lambda row: np.concatenate(
            [
                np.asarray(row["embedding_2s"], dtype=float),
                np.asarray(row["embedding_8s"], dtype=float),
            ]
        ).astype(float),
        axis=1,
    )
    concat_matrix = np.vstack(result["embedding_concat"].to_numpy())
    for index in range(concat_matrix.shape[1]):
        result[f"embedding_{index + 1}"] = concat_matrix[:, index]
    return result.drop(columns=["embedding_2s", "embedding_8s", "embedding_concat"]).reset_index(drop=True)


def get_embedding_columns(embedding_case_df: pd.DataFrame) -> list[str]:
    return [column for column in embedding_case_df.columns if column.startswith("embedding_")]


def load_previous_057_case_table() -> pd.DataFrame:
    if not TRY057_CASE_TABLE_PATH.exists():
        raise FileNotFoundError(f"缺少旧 057 embedding 表: {TRY057_CASE_TABLE_PATH}")
    return pd.read_csv(TRY057_CASE_TABLE_PATH, encoding="utf-8-sig")
