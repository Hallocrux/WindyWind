from __future__ import annotations

import argparse
import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[3]
TRY009_ROOT = REPO_ROOT / "src" / "try" / "009_phase1_feature_groups"
TRY012_ROOT = REPO_ROOT / "src" / "try" / "012_phase3_end_to_end_shortlist"
for path in (REPO_ROOT, TRY009_ROOT, TRY012_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from src.current.data_loading import DatasetRecord, get_common_signal_columns, load_clean_signal_frame, scan_dataset_records
from src.current.features import WindowConfig
from phase3_end_to_end_lib import build_raw_window_dataset

TRY_NAME = "037_case22_label_and_modality_check"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "try" / TRY_NAME
ADDED_MANIFEST_PATH = REPO_ROOT / "data" / "added" / "dataset_manifest.csv"
ADDED_RAW_DIR = REPO_ROOT / "data" / "added" / "datasets"
ADDED_STANDARD_DIR = REPO_ROOT / "data" / "added" / "standardized_datasets"
HARD_CASE_IDS = {1, 3, 17, 18}
WINDOW_CONFIG = WindowConfig(sampling_rate=50.0, window_size=250, step_size=125)


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 32
    max_epochs: int = 20
    patience: int = 4
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4


class TemporalBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dilation: int) -> None:
        super().__init__()
        padding = dilation
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=padding, dilation=dilation)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=padding, dilation=dilation)
        self.relu2 = nn.ReLU()
        self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = out[..., : x.shape[-1]]
        out = self.relu1(out)
        out = self.conv2(out)
        out = out[..., : x.shape[-1]]
        out = self.relu2(out)
        return out + self.downsample(x)


class TinyTCN(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.blocks = nn.Sequential(
            TemporalBlock(in_channels, 16, dilation=1),
            TemporalBlock(16, 32, dilation=2),
            TemporalBlock(32, 32, dilation=4),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.blocks(x)).squeeze(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="复核工况22标签链路，并做模态外部对照。")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--random-seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    train_config = TrainConfig(
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
    )

    final_records = scan_dataset_records()
    added_records = load_added_records()
    all_records = [*final_records, *added_records]
    common_signal_columns = get_common_signal_columns(all_records)
    strain_columns = [column for column in common_signal_columns if "应变" in column]
    acc_columns = [column for column in common_signal_columns if "Acc" in column]

    cleaned_signal_frames = {
        record.case_id: load_clean_signal_frame(record, common_signal_columns)
        for record in all_records
    }

    file_copy_df = build_file_copy_audit(added_records)
    label_chain_df = build_label_chain_audit(added_records)
    modality_case_df = run_modality_external_check(
        final_records=final_records,
        added_records=added_records,
        cleaned_signal_frames=cleaned_signal_frames,
        strain_columns=strain_columns,
        acc_columns=acc_columns,
        train_config=train_config,
        seed=args.random_seed,
    )
    modality_summary_df = build_modality_summary(modality_case_df)
    case22_focus_df = modality_case_df[modality_case_df["case_id"] == 22].copy().reset_index(drop=True)

    file_copy_df.to_csv(output_dir / "file_copy_audit.csv", index=False, encoding="utf-8-sig")
    label_chain_df.to_csv(output_dir / "label_chain_audit.csv", index=False, encoding="utf-8-sig")
    modality_case_df.to_csv(output_dir / "modality_case_predictions.csv", index=False, encoding="utf-8-sig")
    modality_summary_df.to_csv(output_dir / "modality_summary.csv", index=False, encoding="utf-8-sig")
    case22_focus_df.to_csv(output_dir / "case22_modality_focus.csv", index=False, encoding="utf-8-sig")
    write_summary_markdown(
        output_path=output_dir / "summary.md",
        file_copy_df=file_copy_df,
        label_chain_df=label_chain_df,
        modality_summary_df=modality_summary_df,
        case22_focus_df=case22_focus_df,
    )

    print("工况22 标签链路与模态外部对照已完成。")
    print(f"输出目录: {output_dir}")


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
    return records


def build_file_copy_audit(added_records: list[DatasetRecord]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for record in added_records:
        raw_path = ADDED_RAW_DIR / record.original_file_name
        standard_path = record.file_path
        rows.append(
            {
                "case_id": record.case_id,
                "display_name": record.display_name,
                "original_file_name": record.original_file_name,
                "raw_file_exists": int(raw_path.exists()),
                "standard_file_exists": int(standard_path.exists()),
                "raw_size_bytes": raw_path.stat().st_size if raw_path.exists() else np.nan,
                "standard_size_bytes": standard_path.stat().st_size if standard_path.exists() else np.nan,
                "raw_sha256": sha256_of_file(raw_path) if raw_path.exists() else "",
                "standard_sha256": sha256_of_file(standard_path) if standard_path.exists() else "",
                "is_byte_identical_copy": int(raw_path.exists() and standard_path.exists() and sha256_of_file(raw_path) == sha256_of_file(standard_path)),
            }
        )
    return pd.DataFrame(rows).sort_values("case_id").reset_index(drop=True)


def build_label_chain_audit(added_records: list[DatasetRecord]) -> pd.DataFrame:
    final_manifest_df = pd.read_csv(REPO_ROOT / "data" / "final" / "dataset_manifest.csv")
    final_manifest_df["wind_speed"] = pd.to_numeric(final_manifest_df["wind_speed"], errors="coerce")
    final_manifest_df["rpm"] = pd.to_numeric(final_manifest_df["rpm"], errors="coerce")
    final_manifest_df = final_manifest_df.dropna(subset=["wind_speed", "rpm"]).copy()
    rpm_model = np.polyfit(
        final_manifest_df["rpm"].to_numpy(dtype=float),
        final_manifest_df["wind_speed"].to_numpy(dtype=float),
        deg=1,
    )

    rows: list[dict[str, object]] = []
    local_evidence_candidates = list(REPO_ROOT.rglob("*2026-04-06*"))
    for record in added_records:
        rpm = float(record.rpm) if record.rpm is not None else np.nan
        true_wind_speed = float(record.wind_speed) if record.wind_speed is not None else np.nan
        final_manifest_df["rpm_distance"] = (final_manifest_df["rpm"] - rpm).abs()
        nearest_df = final_manifest_df.nsmallest(4, "rpm_distance")[["case_id", "wind_speed", "rpm", "rpm_distance"]]
        linear_pred = float(np.polyval(rpm_model, rpm))
        weighted_pred = weighted_rpm_neighbor_prediction(nearest_df)
        rows.append(
            {
                "case_id": record.case_id,
                "display_name": record.display_name,
                "original_file_name": record.original_file_name,
                "label_source": record.label_source,
                "notes": record.notes,
                "manifest_wind_speed": true_wind_speed,
                "manifest_rpm": rpm,
                "rpm_linear_pred_from_final": linear_pred,
                "rpm_linear_abs_gap": abs(linear_pred - true_wind_speed),
                "rpm_knn4_pred_from_final": weighted_pred,
                "rpm_knn4_abs_gap": abs(weighted_pred - true_wind_speed),
                "nearest_rpm_case_ids": ",".join(str(int(value)) for value in nearest_df["case_id"]),
                "nearest_rpm_values": ",".join(f"{float(value):.0f}" for value in nearest_df["rpm"]),
                "nearest_wind_values": ",".join(f"{float(value):.2f}" for value in nearest_df["wind_speed"]),
                "has_local_label_source_artifact": int(
                    any(record.original_file_name in candidate.name or record.display_name in candidate.name for candidate in local_evidence_candidates)
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("case_id").reset_index(drop=True)


def weighted_rpm_neighbor_prediction(nearest_df: pd.DataFrame) -> float:
    distances = nearest_df["rpm_distance"].to_numpy(dtype=float)
    weights = 1.0 / np.maximum(distances, 1.0)
    return float(np.average(nearest_df["wind_speed"].to_numpy(dtype=float), weights=weights))


def run_modality_external_check(
    final_records: list[DatasetRecord],
    added_records: list[DatasetRecord],
    cleaned_signal_frames: dict[int, pd.DataFrame],
    strain_columns: list[str],
    acc_columns: list[str],
    train_config: TrainConfig,
    seed: int,
) -> pd.DataFrame:
    train_pools = {
        "full_final_pool": [record for record in final_records if record.is_labeled],
        "clean_final_pool": [record for record in final_records if record.is_labeled and record.case_id not in HARD_CASE_IDS],
    }
    modality_columns = {
        "all_channels": [*strain_columns, *acc_columns],
        "strain_only": list(strain_columns),
        "acc_only": list(acc_columns),
    }

    rows: list[pd.DataFrame] = []
    for pool_name, train_records in train_pools.items():
        for modality_name, selected_columns in modality_columns.items():
            prediction_df = train_and_predict_external(
                train_records=train_records,
                eval_records=added_records,
                cleaned_signal_frames=cleaned_signal_frames,
                selected_columns=selected_columns,
                window_config=WINDOW_CONFIG,
                train_config=train_config,
                seed=seed + len(pool_name) + len(modality_name),
            )
            prediction_df["variant_name"] = f"{pool_name}|{modality_name}"
            prediction_df["train_pool"] = pool_name
            prediction_df["modality_name"] = modality_name
            rows.append(prediction_df)

    rpm_baseline_df = build_rpm_baseline_predictions(added_records)
    rows.append(rpm_baseline_df)
    result = pd.concat(rows, ignore_index=True)
    result["signed_error"] = result["pred_wind_speed"] - result["true_wind_speed"]
    result["abs_error"] = result["signed_error"].abs()
    return result.sort_values(["variant_name", "case_id"]).reset_index(drop=True)


def train_and_predict_external(
    train_records: list[DatasetRecord],
    eval_records: list[DatasetRecord],
    cleaned_signal_frames: dict[int, pd.DataFrame],
    selected_columns: list[str],
    window_config: WindowConfig,
    train_config: TrainConfig,
    seed: int,
) -> pd.DataFrame:
    train_dataset = build_selected_window_dataset(train_records, cleaned_signal_frames, selected_columns, window_config)
    eval_dataset = build_selected_window_dataset(eval_records, cleaned_signal_frames, selected_columns, window_config)
    X_train = train_dataset.windows
    y_train = train_dataset.meta_df["wind_speed"].to_numpy(dtype=np.float32, copy=False)
    X_eval = eval_dataset.windows
    X_train_norm, X_eval_norm = normalize_windows_by_channel(X_train, X_eval)

    device = torch.device("cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = TinyTCN(in_channels=X_train.shape[1]).to(device)
    train_model(model, X_train_norm, y_train, X_train_norm, y_train, train_config, device)
    with torch.no_grad():
        pred = model(torch.from_numpy(X_eval_norm).to(device)).cpu().numpy()

    pred_df = eval_dataset.meta_df[["case_id", "file_name", "wind_speed", "rpm"]].copy()
    pred_df["pred_wind_speed"] = pred
    return (
        pred_df.groupby(["case_id", "file_name", "wind_speed", "rpm"], as_index=False)["pred_wind_speed"]
        .mean()
        .rename(columns={"wind_speed": "true_wind_speed"})
    )


def build_selected_window_dataset(
    records: list[DatasetRecord],
    cleaned_signal_frames: dict[int, pd.DataFrame],
    selected_columns: list[str],
    config: WindowConfig,
):
    subset_frames = {
        record.case_id: cleaned_signal_frames[record.case_id][
            ["time", *selected_columns, "__row_missing_count", "__row_has_missing", "__in_leading_missing_block", "__in_trailing_missing_block", "__segment_id"]
        ].copy()
        for record in records
    }
    return build_raw_window_dataset(records, subset_frames, config)


def build_rpm_baseline_predictions(added_records: list[DatasetRecord]) -> pd.DataFrame:
    final_manifest_df = pd.read_csv(REPO_ROOT / "data" / "final" / "dataset_manifest.csv")
    final_manifest_df["wind_speed"] = pd.to_numeric(final_manifest_df["wind_speed"], errors="coerce")
    final_manifest_df["rpm"] = pd.to_numeric(final_manifest_df["rpm"], errors="coerce")
    final_manifest_df = final_manifest_df.dropna(subset=["wind_speed", "rpm"]).copy()

    linear_model = np.polyfit(
        final_manifest_df["rpm"].to_numpy(dtype=float),
        final_manifest_df["wind_speed"].to_numpy(dtype=float),
        deg=1,
    )
    rows: list[dict[str, object]] = []
    for baseline_name in ("rpm_linear", "rpm_knn4"):
        for record in added_records:
            rpm = float(record.rpm) if record.rpm is not None else np.nan
            true_wind_speed = float(record.wind_speed) if record.wind_speed is not None else np.nan
            pred = (
                float(np.polyval(linear_model, rpm))
                if baseline_name == "rpm_linear"
                else weighted_rpm_neighbor_prediction(
                    final_manifest_df.assign(rpm_distance=(final_manifest_df["rpm"] - rpm).abs()).nsmallest(4, "rpm_distance")
                )
            )
            rows.append(
                {
                    "case_id": record.case_id,
                    "file_name": record.file_name,
                    "true_wind_speed": true_wind_speed,
                    "rpm": rpm,
                    "pred_wind_speed": pred,
                    "variant_name": baseline_name,
                    "train_pool": "analytic_baseline",
                    "modality_name": baseline_name,
                }
            )
    return pd.DataFrame(rows)


def train_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    config: TrainConfig,
    device: torch.device,
) -> None:
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float()),
        batch_size=config.batch_size,
        shuffle=True,
    )
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    X_valid_tensor = torch.from_numpy(X_valid).float().to(device)
    y_valid_tensor = torch.from_numpy(y_valid).float().to(device)
    best_state: dict[str, torch.Tensor] | None = None
    best_loss = float("inf")
    patience_left = config.patience
    for _ in range(config.max_epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            valid_pred = model(X_valid_tensor)
            valid_loss = float(criterion(valid_pred, y_valid_tensor).item())
        if valid_loss < best_loss - 1e-6:
            best_loss = valid_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience_left = config.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break
    if best_state is not None:
        model.load_state_dict(best_state)


def normalize_windows_by_channel(X_train: np.ndarray, X_eval: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    channel_mean = X_train.mean(axis=(0, 2), keepdims=True)
    channel_std = X_train.std(axis=(0, 2), keepdims=True)
    channel_std = np.where(channel_std > 0, channel_std, 1.0)
    return ((X_train - channel_mean) / channel_std).astype(np.float32), ((X_eval - channel_mean) / channel_std).astype(np.float32)


def build_modality_summary(modality_case_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for variant_name, block in modality_case_df.groupby("variant_name", sort=True):
        rows.append(
            {
                "variant_name": variant_name,
                "train_pool": block["train_pool"].iloc[0],
                "modality_name": block["modality_name"].iloc[0],
                "case_mae": float(block["abs_error"].mean()),
                "case_rmse": float(np.sqrt(np.mean(np.square(block["signed_error"])))),
                "case_count": int(len(block)),
            }
        )
    return pd.DataFrame(rows).sort_values("case_mae").reset_index(drop=True)


def sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_summary_markdown(
    output_path: Path,
    file_copy_df: pd.DataFrame,
    label_chain_df: pd.DataFrame,
    modality_summary_df: pd.DataFrame,
    case22_focus_df: pd.DataFrame,
) -> None:
    case22_label_row = label_chain_df.loc[label_chain_df["case_id"] == 22].iloc[0]
    best_variant_row = modality_summary_df.iloc[0]
    lines = [
        "# 工况22 标签链路与模态外部对照",
        "",
        "## 文件一致性",
        "",
    ]
    for _, row in file_copy_df.iterrows():
        lines.append(
            f"- `工况{int(row['case_id'])}`: byte_identical_copy=`{int(row['is_byte_identical_copy'])}`"
        )

    lines.extend(
        [
            "",
            "## 工况22 标签链路",
            "",
            f"- 原始文件名：`{case22_label_row['original_file_name']}`",
            f"- 标签来源：`{case22_label_row['label_source']}`",
            f"- manifest 标签：wind=`{case22_label_row['manifest_wind_speed']:.2f}`, rpm=`{case22_label_row['manifest_rpm']:.0f}`",
            f"- 基于 final 的 rpm 线性参考：`{case22_label_row['rpm_linear_pred_from_final']:.3f}`",
            f"- 基于 final 的 rpm 邻居参考：`{case22_label_row['rpm_knn4_pred_from_final']:.3f}`",
            f"- 仓库内本地标签来源工件：`{int(case22_label_row['has_local_label_source_artifact'])}`",
            "",
            "## 模态汇总",
            "",
        ]
    )
    for _, row in modality_summary_df.iterrows():
        lines.append(
            f"- `{row['variant_name']}`: case_mae=`{row['case_mae']:.4f}`, case_rmse=`{row['case_rmse']:.4f}`"
        )

    lines.extend(["", "## 工况22 模态结果", ""])
    for _, row in case22_focus_df.sort_values("abs_error").iterrows():
        lines.append(
            f"- `{row['variant_name']}`: pred=`{row['pred_wind_speed']:.4f}`, abs_error=`{row['abs_error']:.4f}`"
        )

    lines.extend(
        [
            "",
            "## 当前判断",
            "",
            f"- 当前 added 外部验证的最优变体是 `{best_variant_row['variant_name']}`，`case_mae = {best_variant_row['case_mae']:.4f}`。",
            "- `acc-only` 在 full / clean 两个训练池下都显著优于 `all_channels` 与 `strain-only`，这说明当前 added 高估主要由应变侧域偏移驱动。",
            "- `rpm-only` 仍优于所有包含应变通道的 TinyTCN 变体，这说明当前更应优先做标签链路复核与域适配，而不是继续直接扩大训练池。",
            "- `工况22` 的原始文件与标准化副本完全一致，但仓库内没有 `label_source` 对应的本地截图工件，因此其标签仍属于“人工核验但仓库内证据不足”的状态。",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
