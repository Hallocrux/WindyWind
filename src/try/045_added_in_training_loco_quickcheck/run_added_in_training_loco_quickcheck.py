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

from src.current.data_loading import DatasetRecord, get_common_signal_columns, load_clean_signal_frame, scan_dataset_records
from src.current.features import WindowConfig
from phase3_end_to_end_lib import build_raw_window_dataset

TRY_NAME = "045_added_in_training_loco_quickcheck"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "try" / TRY_NAME
ADDED_MANIFEST_PATH = REPO_ROOT / "data" / "added" / "dataset_manifest.csv"
ADDED_DATA_DIR = REPO_ROOT / "data" / "added" / "standardized_datasets"
WINDOW_CONFIGS = {
    "2s": WindowConfig(sampling_rate=50.0, window_size=100, step_size=50),
    "8s": WindowConfig(sampling_rate=50.0, window_size=400, step_size=200),
}


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
    parser = argparse.ArgumentParser(description="added 并入统一训练池后的 2s/8s LOCO quickcheck。")
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
    all_records = sorted([*final_records, *added_records], key=lambda record: record.case_id)
    domain_by_case_id = {
        **{record.case_id: "final" for record in final_records},
        **{record.case_id: "added" for record in added_records},
    }

    common_signal_columns = get_common_signal_columns(all_records)
    cleaned_signal_frames = {
        record.case_id: load_clean_signal_frame(record, common_signal_columns)
        for record in all_records
    }

    prediction_frames: dict[str, pd.DataFrame] = {}
    for index, (window_label, window_config) in enumerate(WINDOW_CONFIGS.items(), start=1):
        prediction_frames[window_label] = run_loco_for_window(
            all_records=all_records,
            cleaned_signal_frames=cleaned_signal_frames,
            window_config=window_config,
            train_config=train_config,
            domain_by_case_id=domain_by_case_id,
            seed=args.random_seed + index * 100,
        )

    case_level_df = merge_predictions(prediction_frames)
    summary_df = build_summary_by_domain(case_level_df)
    write_summary_markdown(output_dir / "summary.md", case_level_df, summary_df)

    case_level_df.to_csv(output_dir / "case_level_predictions.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(output_dir / "summary_by_domain.csv", index=False, encoding="utf-8-sig")

    print("added 并入训练池后的 2s/8s LOCO quickcheck 已完成。")
    print(f"输出目录: {output_dir}")


def load_added_records() -> list[DatasetRecord]:
    manifest_df = pd.read_csv(ADDED_MANIFEST_PATH)
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
                file_path=ADDED_DATA_DIR / f"工况{case_id}.csv",
                wind_speed=wind_speed,
                rpm=rpm,
                is_labeled=wind_speed is not None and rpm is not None,
                original_file_name=str(row["original_file_name"]),
                label_source=str(row["label_source"]),
                notes=str(row["notes"]),
            )
        )
    return [record for record in records if record.is_labeled]


def run_loco_for_window(
    *,
    all_records: list[DatasetRecord],
    cleaned_signal_frames: dict[int, pd.DataFrame],
    window_config: WindowConfig,
    train_config: TrainConfig,
    domain_by_case_id: dict[int, str],
    seed: int,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for holdout in all_records:
        train_records = [record for record in all_records if record.case_id != holdout.case_id]
        pred_df = train_and_predict_holdout(
            train_records=train_records,
            eval_record=holdout,
            cleaned_signal_frames=cleaned_signal_frames,
            window_config=window_config,
            train_config=train_config,
            seed=seed + holdout.case_id,
        )
        pred_df["domain"] = domain_by_case_id[holdout.case_id]
        rows.append(pred_df)
    return pd.concat(rows, ignore_index=True)


def train_and_predict_holdout(
    *,
    train_records: list[DatasetRecord],
    eval_record: DatasetRecord,
    cleaned_signal_frames: dict[int, pd.DataFrame],
    window_config: WindowConfig,
    train_config: TrainConfig,
    seed: int,
) -> pd.DataFrame:
    train_dataset = build_raw_window_dataset(
        train_records,
        {record.case_id: cleaned_signal_frames[record.case_id] for record in train_records},
        window_config,
    )
    eval_dataset = build_raw_window_dataset(
        [eval_record],
        {eval_record.case_id: cleaned_signal_frames[eval_record.case_id]},
        window_config,
    )
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


def merge_predictions(prediction_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    base: pd.DataFrame | None = None
    for label, df in prediction_frames.items():
        block = df.rename(columns={"pred_wind_speed": f"pred_{label}"})
        if base is None:
            base = block.copy()
        else:
            base = base.merge(block[["case_id", f"pred_{label}"]], on="case_id", how="left")
    assert base is not None

    base["pred_2s_8s_fusion"] = (base["pred_2s"] + base["pred_8s"]) / 2.0
    for model_name in ("2s", "8s", "2s_8s_fusion"):
        pred_column = f"pred_{model_name}"
        signed_error_column = f"signed_error_{model_name}"
        abs_error_column = f"abs_error_{model_name}"
        base[signed_error_column] = base[pred_column] - base["true_wind_speed"]
        base[abs_error_column] = base[signed_error_column].abs()
    return base.sort_values(["domain", "case_id"]).reset_index(drop=True)


def build_summary_by_domain(case_level_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for domain_name, subset in (
        ("final", case_level_df.loc[case_level_df["domain"] == "final"].copy()),
        ("added", case_level_df.loc[case_level_df["domain"] == "added"].copy()),
        ("all_labeled", case_level_df.copy()),
    ):
        for model_name in ("2s", "8s", "2s_8s_fusion"):
            signed_error = subset[f"signed_error_{model_name}"].to_numpy(dtype=float, copy=False)
            abs_error = subset[f"abs_error_{model_name}"].to_numpy(dtype=float, copy=False)
            rows.append(
                {
                    "domain": domain_name,
                    "model_name": model_name,
                    "case_mae": float(np.mean(abs_error)),
                    "case_rmse": float(np.sqrt(np.mean(np.square(signed_error)))),
                    "case_count": int(len(subset)),
                }
            )
    return pd.DataFrame(rows).sort_values(["domain", "case_mae", "model_name"]).reset_index(drop=True)


def write_summary_markdown(output_path: Path, case_level_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    lines = [
        "# added 并入训练池后的 2s/8s LOCO quickcheck",
        "",
        f"- `final` 带标签工况数：`{int((case_level_df['domain'] == 'final').sum())}`",
        f"- `added` 带标签工况数：`{int((case_level_df['domain'] == 'added').sum())}`",
        f"- `all_labeled` 工况数：`{int(len(case_level_df))}`",
        "",
        "## 三桶汇总",
        "",
    ]
    for domain_name in ("final", "added", "all_labeled"):
        lines.append(f"### {domain_name}")
        lines.append("")
        block = summary_df.loc[summary_df["domain"] == domain_name].copy()
        for _, row in block.iterrows():
            lines.append(
                f"- `{row['model_name']}`: case_mae=`{row['case_mae']:.4f}`, case_rmse=`{row['case_rmse']:.4f}`, case_count=`{int(row['case_count'])}`"
            )
        lines.append("")

    lines.extend(["## 每工况预测", ""])
    for _, row in case_level_df.iterrows():
        lines.append(
            f"- `{row['domain']} | 工况{int(row['case_id'])}`: true=`{row['true_wind_speed']:.4f}`, pred_2s=`{row['pred_2s']:.4f}`, pred_8s=`{row['pred_8s']:.4f}`, pred_fusion=`{row['pred_2s_8s_fusion']:.4f}`"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
