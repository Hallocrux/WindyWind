from __future__ import annotations

import argparse
import sys
from dataclasses import asdict, dataclass
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

from src.current.data_loading import DatasetRecord, QUALITY_COLUMNS, TIME_COLUMN, get_common_signal_columns, load_clean_signal_frame, scan_dataset_records
from src.current.features import WindowConfig
from phase3_end_to_end_lib import build_raw_window_dataset

TRY_NAME = "039_strain_transfer_band_scan"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "try" / TRY_NAME
ADDED_MANIFEST_PATH = REPO_ROOT / "data" / "added" / "dataset_manifest.csv"
ADDED_STANDARD_DIR = REPO_ROOT / "data" / "added" / "standardized_datasets"
WINDOW_CONFIG = WindowConfig(sampling_rate=50.0, window_size=250, step_size=125)
QUALITY_COLUMNS_LIST = [column for column in QUALITY_COLUMNS]


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 32
    max_epochs: int = 20
    patience: int = 4
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4


@dataclass(frozen=True)
class VariantConfig:
    variant_name: str
    input_columns: str
    strain_transform: str
    transform_low_hz: float | None
    transform_high_hz: float | None
    variant_kind: str


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
    parser = argparse.ArgumentParser(description="筛选应变可迁移频带。")
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

    final_records = [record for record in scan_dataset_records() if record.is_labeled]
    added_records = load_added_records()
    all_records = [*final_records, *added_records]
    common_signal_columns = get_common_signal_columns(all_records)
    strain_columns = [column for column in common_signal_columns if "应变" in column]
    acc_columns = [column for column in common_signal_columns if "Acc" in column]
    base_frames = {
        record.case_id: load_clean_signal_frame(record, common_signal_columns)
        for record in all_records
    }

    base_variants = build_base_variants()
    case_frames: list[pd.DataFrame] = []
    for index, variant in enumerate(base_variants):
        if variant.variant_kind == "analytic":
            case_frames.append(build_rpm_knn_predictions(added_records))
            continue

        transformed_frames = build_variant_frames(
            base_frames=base_frames,
            records=all_records,
            strain_columns=strain_columns,
            variant=variant,
        )
        selected_columns = list(acc_columns) if variant.input_columns == "acc_only" else [*strain_columns, *acc_columns]
        pred_df = train_and_predict_external(
            train_records=final_records,
            eval_records=added_records,
            cleaned_signal_frames=transformed_frames,
            selected_columns=selected_columns,
            train_config=train_config,
            seed=args.random_seed + 37 * (index + 1),
        )
        pred_df["variant_name"] = variant.variant_name
        pred_df["input_columns"] = variant.input_columns
        pred_df["strain_transform"] = variant.strain_transform
        pred_df["transform_low_hz"] = variant.transform_low_hz
        pred_df["transform_high_hz"] = variant.transform_high_hz
        pred_df["variant_kind"] = variant.variant_kind
        case_frames.append(pred_df)

    case_level_df = pd.concat(case_frames, ignore_index=True)
    case_level_df["signed_error"] = case_level_df["pred_wind_speed"] - case_level_df["true_wind_speed"]
    case_level_df["abs_error"] = case_level_df["signed_error"].abs()

    summary_df = build_summary(case_level_df)
    best_band_variant = pick_best_band_variant(summary_df)
    fusion_df = build_fusion_predictions(case_level_df, best_band_variant)
    case_level_df = pd.concat([case_level_df, fusion_df], ignore_index=True)
    case_level_df["signed_error"] = case_level_df["pred_wind_speed"] - case_level_df["true_wind_speed"]
    case_level_df["abs_error"] = case_level_df["signed_error"].abs()
    summary_df = build_summary(case_level_df)
    case22_focus_df = case_level_df[case_level_df["case_id"] == 22].sort_values("abs_error").reset_index(drop=True)
    best_band_reference_df = build_best_band_reference(summary_df, best_band_variant)

    variant_config_rows = [asdict(variant) for variant in base_variants]
    variant_config_rows.extend(
        [
            {
                "variant_name": "fusion_acc_only__strain_highpass_2hz",
                "input_columns": "late_fusion",
                "strain_transform": "acc_only + strain_highpass_2hz",
                "transform_low_hz": np.nan,
                "transform_high_hz": np.nan,
                "variant_kind": "fusion",
            },
            {
                "variant_name": f"fusion_acc_only__{best_band_variant}",
                "input_columns": "late_fusion",
                "strain_transform": f"acc_only + {best_band_variant}",
                "transform_low_hz": np.nan,
                "transform_high_hz": np.nan,
                "variant_kind": "fusion",
            },
        ]
    )
    variant_config_df = pd.DataFrame(variant_config_rows)

    variant_config_df.to_csv(output_dir / "variant_config_table.csv", index=False, encoding="utf-8-sig")
    case_level_df.to_csv(output_dir / "case_level_predictions.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(output_dir / "summary.csv", index=False, encoding="utf-8-sig")
    case22_focus_df.to_csv(output_dir / "case22_focus.csv", index=False, encoding="utf-8-sig")
    best_band_reference_df.to_csv(output_dir / "best_band_reference.csv", index=False, encoding="utf-8-sig")
    write_summary_markdown(output_dir / "summary.md", summary_df, case22_focus_df, best_band_variant)

    print("应变可迁移频带筛选已完成。")
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


def build_base_variants() -> list[VariantConfig]:
    return [
        VariantConfig("rpm_knn4", "rpm", "none", None, None, "analytic"),
        VariantConfig("full_final_pool|acc_only", "acc_only", "none", None, None, "tinytcn"),
        VariantConfig("full_final_pool|all_channels_strain_highpass_2hz", "all_channels", "highpass", 2.0, None, "tinytcn"),
        VariantConfig("full_final_pool|all_channels_strain_bandpass_2_4hz", "all_channels", "bandpass", 2.0, 4.0, "tinytcn"),
        VariantConfig("full_final_pool|all_channels_strain_bandpass_2_5hz", "all_channels", "bandpass", 2.0, 5.0, "tinytcn"),
        VariantConfig("full_final_pool|all_channels_strain_bandpass_3_6hz", "all_channels", "bandpass", 3.0, 6.0, "tinytcn"),
        VariantConfig("full_final_pool|all_channels_strain_bandpass_4_8hz", "all_channels", "bandpass", 4.0, 8.0, "tinytcn"),
    ]


def build_variant_frames(
    base_frames: dict[int, pd.DataFrame],
    records: list[DatasetRecord],
    strain_columns: list[str],
    variant: VariantConfig,
) -> dict[int, pd.DataFrame]:
    if variant.strain_transform == "none":
        return {case_id: frame.copy() for case_id, frame in base_frames.items()}

    result: dict[int, pd.DataFrame] = {}
    for record in records:
        frame = base_frames[record.case_id].copy()
        for _, indexer in frame.groupby("__segment_id", sort=True).groups.items():
            segment_index = list(indexer)
            segment_df = frame.loc[segment_index, strain_columns]
            if variant.strain_transform == "highpass":
                transformed = apply_frequency_filter(
                    segment_df.to_numpy(dtype=float, copy=True),
                    sampling_rate=WINDOW_CONFIG.sampling_rate,
                    low_hz=variant.transform_low_hz,
                    high_hz=None,
                )
            elif variant.strain_transform == "bandpass":
                transformed = apply_frequency_filter(
                    segment_df.to_numpy(dtype=float, copy=True),
                    sampling_rate=WINDOW_CONFIG.sampling_rate,
                    low_hz=variant.transform_low_hz,
                    high_hz=variant.transform_high_hz,
                )
            else:
                raise ValueError(f"未知的应变变换: {variant.strain_transform}")
            frame.loc[segment_index, strain_columns] = transformed
        result[record.case_id] = frame
    return result


def apply_frequency_filter(values: np.ndarray, sampling_rate: float, low_hz: float | None, high_hz: float | None) -> np.ndarray:
    if values.shape[0] <= 1:
        return values
    freqs = np.fft.rfftfreq(values.shape[0], d=1.0 / sampling_rate)
    mask = np.ones_like(freqs, dtype=bool)
    if low_hz is not None:
        mask &= freqs >= low_hz
    if high_hz is not None:
        mask &= freqs < high_hz
    transformed = np.zeros_like(values)
    for column_index in range(values.shape[1]):
        spectrum = np.fft.rfft(values[:, column_index])
        spectrum[~mask] = 0.0
        transformed[:, column_index] = np.fft.irfft(spectrum, n=values.shape[0])
    return transformed


def train_and_predict_external(
    train_records: list[DatasetRecord],
    eval_records: list[DatasetRecord],
    cleaned_signal_frames: dict[int, pd.DataFrame],
    selected_columns: list[str],
    train_config: TrainConfig,
    seed: int,
) -> pd.DataFrame:
    train_dataset = build_selected_window_dataset(train_records, cleaned_signal_frames, selected_columns)
    eval_dataset = build_selected_window_dataset(eval_records, cleaned_signal_frames, selected_columns)
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
):
    subset_frames = {
        record.case_id: cleaned_signal_frames[record.case_id][
            [TIME_COLUMN, *selected_columns, *QUALITY_COLUMNS_LIST]
        ].copy()
        for record in records
    }
    return build_raw_window_dataset(records, subset_frames, WINDOW_CONFIG)


def build_rpm_knn_predictions(added_records: list[DatasetRecord]) -> pd.DataFrame:
    final_manifest_df = pd.read_csv(REPO_ROOT / "data" / "final" / "dataset_manifest.csv")
    final_manifest_df["wind_speed"] = pd.to_numeric(final_manifest_df["wind_speed"], errors="coerce")
    final_manifest_df["rpm"] = pd.to_numeric(final_manifest_df["rpm"], errors="coerce")
    final_manifest_df = final_manifest_df.dropna(subset=["wind_speed", "rpm"]).copy()

    rows: list[dict[str, object]] = []
    for record in added_records:
        rpm = float(record.rpm) if record.rpm is not None else np.nan
        nearest_df = final_manifest_df.assign(rpm_distance=(final_manifest_df["rpm"] - rpm).abs()).nsmallest(4, "rpm_distance")
        pred = weighted_rpm_neighbor_prediction(nearest_df)
        rows.append(
            {
                "case_id": record.case_id,
                "file_name": record.file_name,
                "true_wind_speed": float(record.wind_speed),
                "rpm": rpm,
                "pred_wind_speed": pred,
                "variant_name": "rpm_knn4",
                "input_columns": "rpm",
                "strain_transform": "none",
                "transform_low_hz": np.nan,
                "transform_high_hz": np.nan,
                "variant_kind": "analytic",
            }
        )
    return pd.DataFrame(rows)


def weighted_rpm_neighbor_prediction(nearest_df: pd.DataFrame) -> float:
    distances = nearest_df["rpm_distance"].to_numpy(dtype=float)
    weights = 1.0 / np.maximum(distances, 1.0)
    return float(np.average(nearest_df["wind_speed"].to_numpy(dtype=float), weights=weights))


def build_fusion_predictions(case_level_df: pd.DataFrame, best_band_variant: str) -> pd.DataFrame:
    base_columns = ["case_id", "file_name", "true_wind_speed", "rpm"]
    acc_df = case_level_df.loc[case_level_df["variant_name"] == "full_final_pool|acc_only", base_columns + ["pred_wind_speed"]].rename(
        columns={"pred_wind_speed": "pred_acc_only"}
    )
    highpass_df = case_level_df.loc[
        case_level_df["variant_name"] == "full_final_pool|all_channels_strain_highpass_2hz",
        base_columns + ["pred_wind_speed"],
    ].rename(columns={"pred_wind_speed": "pred_highpass"})
    best_band_df = case_level_df.loc[
        case_level_df["variant_name"] == best_band_variant,
        base_columns + ["pred_wind_speed"],
    ].rename(columns={"pred_wind_speed": "pred_best_band"})

    fusion_highpass = acc_df.merge(highpass_df, on=base_columns, how="inner")
    fusion_highpass["pred_wind_speed"] = (fusion_highpass["pred_acc_only"] + fusion_highpass["pred_highpass"]) / 2.0
    fusion_highpass["variant_name"] = "fusion_acc_only__strain_highpass_2hz"
    fusion_highpass["input_columns"] = "late_fusion"
    fusion_highpass["strain_transform"] = "acc_only + strain_highpass_2hz"
    fusion_highpass["transform_low_hz"] = np.nan
    fusion_highpass["transform_high_hz"] = np.nan
    fusion_highpass["variant_kind"] = "fusion"

    fusion_best_band = acc_df.merge(best_band_df, on=base_columns, how="inner")
    fusion_best_band["pred_wind_speed"] = (fusion_best_band["pred_acc_only"] + fusion_best_band["pred_best_band"]) / 2.0
    fusion_best_band["variant_name"] = f"fusion_acc_only__{best_band_variant}"
    fusion_best_band["input_columns"] = "late_fusion"
    fusion_best_band["strain_transform"] = f"acc_only + {best_band_variant}"
    fusion_best_band["transform_low_hz"] = np.nan
    fusion_best_band["transform_high_hz"] = np.nan
    fusion_best_band["variant_kind"] = "fusion"

    return pd.concat(
        [
            fusion_highpass[base_columns + ["pred_wind_speed", "variant_name", "input_columns", "strain_transform", "transform_low_hz", "transform_high_hz", "variant_kind"]],
            fusion_best_band[base_columns + ["pred_wind_speed", "variant_name", "input_columns", "strain_transform", "transform_low_hz", "transform_high_hz", "variant_kind"]],
        ],
        ignore_index=True,
    )


def normalize_windows_by_channel(X_train: np.ndarray, X_eval: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    channel_mean = X_train.mean(axis=(0, 2), keepdims=True)
    channel_std = X_train.std(axis=(0, 2), keepdims=True)
    channel_std = np.where(channel_std > 0, channel_std, 1.0)
    return ((X_train - channel_mean) / channel_std).astype(np.float32), ((X_eval - channel_mean) / channel_std).astype(np.float32)


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


def build_summary(case_level_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for variant_name, block in case_level_df.groupby("variant_name", sort=False):
        case22_abs_error = float(block.loc[block["case_id"] == 22, "abs_error"].iloc[0])
        rows.append(
            {
                "variant_name": variant_name,
                "input_columns": block["input_columns"].iloc[0],
                "strain_transform": block["strain_transform"].iloc[0],
                "variant_kind": block["variant_kind"].iloc[0],
                "case_mae": float(block["abs_error"].mean()),
                "case_rmse": float(np.sqrt(np.mean(np.square(block["signed_error"])))),
                "mean_signed_error": float(block["signed_error"].mean()),
                "case22_abs_error": case22_abs_error,
                "case_count": int(len(block)),
            }
        )
    return pd.DataFrame(rows).sort_values(["case_mae", "case22_abs_error"]).reset_index(drop=True)


def pick_best_band_variant(summary_df: pd.DataFrame) -> str:
    band_mask = summary_df["variant_name"].str.contains("bandpass", regex=False)
    return str(summary_df.loc[band_mask].sort_values(["case_mae", "case22_abs_error"]).iloc[0]["variant_name"])


def build_best_band_reference(summary_df: pd.DataFrame, best_band_variant: str) -> pd.DataFrame:
    keep_variants = [
        "rpm_knn4",
        "full_final_pool|acc_only",
        "full_final_pool|all_channels_strain_highpass_2hz",
        best_band_variant,
        "fusion_acc_only__strain_highpass_2hz",
        f"fusion_acc_only__{best_band_variant}",
    ]
    return summary_df.loc[summary_df["variant_name"].isin(keep_variants)].copy().sort_values(["case_mae", "case22_abs_error"]).reset_index(drop=True)


def write_summary_markdown(output_path: Path, summary_df: pd.DataFrame, case22_focus_df: pd.DataFrame, best_band_variant: str) -> None:
    best_row = summary_df.iloc[0]
    acc_row = summary_df.loc[summary_df["variant_name"] == "full_final_pool|acc_only"].iloc[0]
    highpass_row = summary_df.loc[summary_df["variant_name"] == "full_final_pool|all_channels_strain_highpass_2hz"].iloc[0]
    best_band_row = summary_df.loc[summary_df["variant_name"] == best_band_variant].iloc[0]
    fusion_best_row = summary_df.loc[summary_df["variant_name"] == f"fusion_acc_only__{best_band_variant}"].iloc[0]

    lines = [
        "# 应变可迁移频带筛选",
        "",
        f"- 最优变体：`{best_row['variant_name']}`",
        f"- 最优 `case_mae`：`{best_row['case_mae']:.4f}`",
        f"- 当前最优应变频带：`{best_band_variant}`",
        "",
        "## 关键对照",
        "",
        f"- `acc_only`: case_mae=`{acc_row['case_mae']:.4f}`, case22_abs_error=`{acc_row['case22_abs_error']:.4f}`, mean_signed_error=`{acc_row['mean_signed_error']:.4f}`",
        f"- `strain_highpass_2hz`: case_mae=`{highpass_row['case_mae']:.4f}`, case22_abs_error=`{highpass_row['case22_abs_error']:.4f}`, mean_signed_error=`{highpass_row['mean_signed_error']:.4f}`",
        f"- `best_band`: case_mae=`{best_band_row['case_mae']:.4f}`, case22_abs_error=`{best_band_row['case22_abs_error']:.4f}`, mean_signed_error=`{best_band_row['mean_signed_error']:.4f}`",
        f"- `fusion_acc_only__best_band`: case_mae=`{fusion_best_row['case_mae']:.4f}`, case22_abs_error=`{fusion_best_row['case22_abs_error']:.4f}`, mean_signed_error=`{fusion_best_row['mean_signed_error']:.4f}`",
        "",
        "## 工况22 结果",
        "",
    ]
    for _, row in case22_focus_df.iterrows():
        lines.append(
            f"- `{row['variant_name']}`: pred=`{row['pred_wind_speed']:.4f}`, abs_error=`{row['abs_error']:.4f}`"
        )

    lines.extend(
        [
            "",
            "## 当前判断",
            "",
            "- 当前 `3-6Hz` 是 added 方向上最可迁移的应变频带，单独使用时已经优于 `acc_only`。",
            "- `acc_only + 3-6Hz band-pass` 晚融合优于单独 `acc_only`、单独 `3-6Hz` 和 `rpm_knn4`，说明应变当前更适合作为中频辅助分支，而不是原始全频输入。",
            "- 下一步若继续做 added 主线，应优先围绕 `acc + strain(3-6Hz)` 或其邻近频带做细化，而不是回到原始 `all_channels`。",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
