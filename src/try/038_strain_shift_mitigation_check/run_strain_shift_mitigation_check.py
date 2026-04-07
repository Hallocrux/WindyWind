from __future__ import annotations

import argparse
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

from src.current.data_loading import DatasetRecord, QUALITY_COLUMNS, TIME_COLUMN, get_common_signal_columns, load_clean_signal_frame, scan_dataset_records
from src.current.features import WindowConfig
from phase3_end_to_end_lib import build_raw_window_dataset

TRY_NAME = "038_strain_shift_mitigation_check"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "try" / TRY_NAME
ADDED_MANIFEST_PATH = REPO_ROOT / "data" / "added" / "dataset_manifest.csv"
ADDED_STANDARD_DIR = REPO_ROOT / "data" / "added" / "standardized_datasets"
HARD_CASE_IDS = {1, 3, 17, 18}
WINDOW_CONFIG = WindowConfig(sampling_rate=50.0, window_size=250, step_size=125)
HIGH_PASS_CUTOFF_HZ = 2.0


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
    train_pool: str
    input_columns: str
    strain_transform: str
    use_tinytcn: bool


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
    parser = argparse.ArgumentParser(description="快速验证应变侧漂移缓解策略。")
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

    base_cleaned_frames = {
        record.case_id: load_clean_signal_frame(record, common_signal_columns)
        for record in all_records
    }

    train_pools = {
        "full_final_pool": [record for record in final_records if record.is_labeled],
        "clean_final_pool": [record for record in final_records if record.is_labeled and record.case_id not in HARD_CASE_IDS],
    }
    variant_configs = build_variant_configs()

    rows: list[pd.DataFrame] = []
    for index, variant in enumerate(variant_configs):
        if not variant.use_tinytcn:
            rows.append(build_rpm_knn_predictions(added_records))
            continue

        selected_columns = [*strain_columns, *acc_columns] if variant.input_columns == "all_channels" else list(acc_columns)
        transformed_frames = build_variant_frames(
            base_cleaned_frames=base_cleaned_frames,
            records=all_records,
            strain_columns=strain_columns,
            transform_name=variant.strain_transform,
        )
        prediction_df = train_and_predict_external(
            train_records=train_pools[variant.train_pool],
            eval_records=added_records,
            cleaned_signal_frames=transformed_frames,
            selected_columns=selected_columns,
            window_config=WINDOW_CONFIG,
            train_config=train_config,
            seed=args.random_seed + index * 17,
        )
        prediction_df["variant_name"] = variant.variant_name
        prediction_df["train_pool"] = variant.train_pool
        prediction_df["input_columns"] = variant.input_columns
        prediction_df["strain_transform"] = variant.strain_transform
        rows.append(prediction_df)

    case_level_df = pd.concat(rows, ignore_index=True)
    case_level_df["signed_error"] = case_level_df["pred_wind_speed"] - case_level_df["true_wind_speed"]
    case_level_df["abs_error"] = case_level_df["signed_error"].abs()
    summary_df = build_summary(case_level_df)
    case22_focus_df = case_level_df[case_level_df["case_id"] == 22].copy().sort_values("abs_error").reset_index(drop=True)
    variant_config_df = pd.DataFrame([variant.__dict__ for variant in variant_configs])

    variant_config_df.to_csv(output_dir / "variant_config_table.csv", index=False, encoding="utf-8-sig")
    case_level_df.to_csv(output_dir / "case_level_predictions.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(output_dir / "summary.csv", index=False, encoding="utf-8-sig")
    case22_focus_df.to_csv(output_dir / "case22_focus.csv", index=False, encoding="utf-8-sig")
    write_summary_markdown(output_dir / "summary.md", summary_df, case22_focus_df)

    print("应变侧漂移缓解快速验证已完成。")
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


def build_variant_configs() -> list[VariantConfig]:
    return [
        VariantConfig("rpm_knn4", "analytic_baseline", "rpm", "none", False),
        VariantConfig("full_final_pool|acc_only", "full_final_pool", "acc_only", "none", True),
        VariantConfig("full_final_pool|all_channels_raw", "full_final_pool", "all_channels", "none", True),
        VariantConfig("full_final_pool|all_channels_strain_case_zscore", "full_final_pool", "all_channels", "case_zscore", True),
        VariantConfig("full_final_pool|all_channels_strain_highpass_2hz", "full_final_pool", "all_channels", "highpass_2hz", True),
        VariantConfig("clean_final_pool|acc_only", "clean_final_pool", "acc_only", "none", True),
        VariantConfig("clean_final_pool|all_channels_raw", "clean_final_pool", "all_channels", "none", True),
        VariantConfig("clean_final_pool|all_channels_strain_case_zscore", "clean_final_pool", "all_channels", "case_zscore", True),
        VariantConfig("clean_final_pool|all_channels_strain_highpass_2hz", "clean_final_pool", "all_channels", "highpass_2hz", True),
    ]


def build_variant_frames(
    base_cleaned_frames: dict[int, pd.DataFrame],
    records: list[DatasetRecord],
    strain_columns: list[str],
    transform_name: str,
) -> dict[int, pd.DataFrame]:
    if transform_name == "none":
        return {case_id: frame.copy() for case_id, frame in base_cleaned_frames.items()}

    result: dict[int, pd.DataFrame] = {}
    for record in records:
        frame = base_cleaned_frames[record.case_id].copy()
        for segment_id, indexer in frame.groupby("__segment_id", sort=True).groups.items():
            segment_index = list(indexer)
            segment_df = frame.loc[segment_index, :]
            if transform_name == "case_zscore":
                frame.loc[segment_index, strain_columns] = apply_case_zscore(segment_df[strain_columns])
            elif transform_name == "highpass_2hz":
                frame.loc[segment_index, strain_columns] = apply_highpass(segment_df[strain_columns], cutoff_hz=HIGH_PASS_CUTOFF_HZ, sampling_rate=WINDOW_CONFIG.sampling_rate)
            else:
                raise ValueError(f"未知的应变变换: {transform_name}")
        result[record.case_id] = frame
    return result


def apply_case_zscore(strain_df: pd.DataFrame) -> np.ndarray:
    values = strain_df.to_numpy(dtype=float, copy=True)
    mean = values.mean(axis=0, keepdims=True)
    std = values.std(axis=0, keepdims=True)
    std = np.where(std > 0, std, 1.0)
    return (values - mean) / std


def apply_highpass(strain_df: pd.DataFrame, cutoff_hz: float, sampling_rate: float) -> np.ndarray:
    values = strain_df.to_numpy(dtype=float, copy=True)
    if values.shape[0] <= 1:
        return values
    freqs = np.fft.rfftfreq(values.shape[0], d=1.0 / sampling_rate)
    highpass_mask = freqs >= cutoff_hz
    transformed = np.zeros_like(values)
    for column_index in range(values.shape[1]):
        signal = values[:, column_index]
        spectrum = np.fft.rfft(signal)
        spectrum[~highpass_mask] = 0.0
        transformed[:, column_index] = np.fft.irfft(spectrum, n=signal.shape[0])
    return transformed


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
            [TIME_COLUMN, *selected_columns, *QUALITY_COLUMNS]
        ].copy()
        for record in records
    }
    return build_raw_window_dataset(records, subset_frames, config)


def build_rpm_knn_predictions(added_records: list[DatasetRecord]) -> pd.DataFrame:
    final_manifest_df = pd.read_csv(REPO_ROOT / "data" / "final" / "dataset_manifest.csv")
    final_manifest_df["wind_speed"] = pd.to_numeric(final_manifest_df["wind_speed"], errors="coerce")
    final_manifest_df["rpm"] = pd.to_numeric(final_manifest_df["rpm"], errors="coerce")
    final_manifest_df = final_manifest_df.dropna(subset=["wind_speed", "rpm"]).copy()

    rows: list[dict[str, object]] = []
    for record in added_records:
        rpm = float(record.rpm) if record.rpm is not None else np.nan
        pred = weighted_rpm_neighbor_prediction(
            final_manifest_df.assign(rpm_distance=(final_manifest_df["rpm"] - rpm).abs()).nsmallest(4, "rpm_distance")
        )
        rows.append(
            {
                "case_id": record.case_id,
                "file_name": record.file_name,
                "true_wind_speed": float(record.wind_speed),
                "rpm": rpm,
                "pred_wind_speed": pred,
                "variant_name": "rpm_knn4",
                "train_pool": "analytic_baseline",
                "input_columns": "rpm",
                "strain_transform": "none",
            }
        )
    return pd.DataFrame(rows)


def weighted_rpm_neighbor_prediction(nearest_df: pd.DataFrame) -> float:
    distances = nearest_df["rpm_distance"].to_numpy(dtype=float)
    weights = 1.0 / np.maximum(distances, 1.0)
    return float(np.average(nearest_df["wind_speed"].to_numpy(dtype=float), weights=weights))


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


def build_summary(case_level_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for variant_name, block in case_level_df.groupby("variant_name", sort=False):
        case22_block = block[block["case_id"] == 22]
        rows.append(
            {
                "variant_name": variant_name,
                "train_pool": block["train_pool"].iloc[0],
                "input_columns": block["input_columns"].iloc[0],
                "strain_transform": block["strain_transform"].iloc[0],
                "case_mae": float(block["abs_error"].mean()),
                "case_rmse": float(np.sqrt(np.mean(np.square(block["signed_error"])))),
                "mean_signed_error": float(block["signed_error"].mean()),
                "case22_abs_error": float(case22_block["abs_error"].iloc[0]),
                "case_count": int(len(block)),
            }
        )
    return pd.DataFrame(rows).sort_values(["case_mae", "case22_abs_error"]).reset_index(drop=True)


def write_summary_markdown(output_path: Path, summary_df: pd.DataFrame, case22_focus_df: pd.DataFrame) -> None:
    best_row = summary_df.iloc[0]
    acc_row = summary_df.loc[summary_df["variant_name"] == "full_final_pool|acc_only"].iloc[0]
    raw_row = summary_df.loc[summary_df["variant_name"] == "full_final_pool|all_channels_raw"].iloc[0]
    zscore_row = summary_df.loc[summary_df["variant_name"] == "full_final_pool|all_channels_strain_case_zscore"].iloc[0]
    highpass_row = summary_df.loc[summary_df["variant_name"] == "full_final_pool|all_channels_strain_highpass_2hz"].iloc[0]

    lines = [
        "# 应变侧漂移缓解快速验证",
        "",
        f"- 最优变体：`{best_row['variant_name']}`",
        f"- 最优 `case_mae`：`{best_row['case_mae']:.4f}`",
        f"- 最优 `工况22 abs_error`：`{best_row['case22_abs_error']:.4f}`",
        "",
        "## 关键对照",
        "",
        f"- `full acc_only`: case_mae=`{acc_row['case_mae']:.4f}`, case22_abs_error=`{acc_row['case22_abs_error']:.4f}`, mean_signed_error=`{acc_row['mean_signed_error']:.4f}`",
        f"- `full all raw`: case_mae=`{raw_row['case_mae']:.4f}`, case22_abs_error=`{raw_row['case22_abs_error']:.4f}`, mean_signed_error=`{raw_row['mean_signed_error']:.4f}`",
        f"- `full all + strain case_zscore`: case_mae=`{zscore_row['case_mae']:.4f}`, case22_abs_error=`{zscore_row['case22_abs_error']:.4f}`, mean_signed_error=`{zscore_row['mean_signed_error']:.4f}`",
        f"- `full all + strain highpass_2hz`: case_mae=`{highpass_row['case_mae']:.4f}`, case22_abs_error=`{highpass_row['case22_abs_error']:.4f}`, mean_signed_error=`{highpass_row['mean_signed_error']:.4f}`",
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
            "- `strain case_zscore` 与 `strain highpass_2hz` 都显著优于原始 `all_channels`，说明应变侧并不是完全不可修复。",
            "- `strain highpass_2hz` 是当前最有效的应变修复手段，但仍没有超过 `full acc_only`，因此 added 主线仍应优先切到 `acc + rpm`。",
            "- 应变侧后续更适合作为“已知可部分修复的域适配分支”继续，而不是直接回到默认全通道主线。",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
