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

TRY_NAME = "035_added_validation_with_full_final_pool"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "try" / TRY_NAME
ADDED_MANIFEST_PATH = REPO_ROOT / "data" / "added" / "dataset_manifest.csv"
ADDED_DATA_DIR = REPO_ROOT / "data" / "added" / "standardized_datasets"
WINDOW_CONFIGS = {
    "2s": WindowConfig(sampling_rate=50.0, window_size=100, step_size=50),
    "5s": WindowConfig(sampling_rate=50.0, window_size=250, step_size=125),
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
    parser = argparse.ArgumentParser(description="用 full final pool 对 added 数据做外部验证。")
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
    common_signal_columns = get_common_signal_columns([*final_records, *added_records])
    cleaned_signal_frames = {
        record.case_id: load_clean_signal_frame(record, common_signal_columns)
        for record in [*final_records, *added_records]
    }

    train_records = [record for record in final_records if record.is_labeled]
    prediction_frames = {}
    for label, window_config in WINDOW_CONFIGS.items():
        prediction_frames[label] = train_and_predict_external(
            train_records=train_records,
            eval_records=added_records,
            cleaned_signal_frames=cleaned_signal_frames,
            window_config=window_config,
            train_config=train_config,
            seed=args.random_seed + len(label),
        )

    added_df = merge_window_predictions(prediction_frames)
    summary_df = build_summary(added_df)

    added_df.to_csv(output_dir / "added_case_predictions.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(output_dir / "summary.csv", index=False, encoding="utf-8-sig")
    write_summary_markdown(output_dir / "summary.md", added_df, summary_df)

    print("added 外部验证（包含难工况训练池）已完成。")
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
                file_path=ADDED_DATA_DIR / f"工况{case_id}.csv",
                wind_speed=float(row["wind_speed"]) if not pd.isna(row["wind_speed"]) else None,
                rpm=float(row["rpm"]) if not pd.isna(row["rpm"]) else None,
                is_labeled=not pd.isna(row["wind_speed"]) and not pd.isna(row["rpm"]),
                original_file_name=str(row["original_file_name"]),
                label_source=str(row["label_source"]),
                notes=str(row["notes"]),
            )
        )
    return records


def train_and_predict_external(
    train_records: list[DatasetRecord],
    eval_records: list[DatasetRecord],
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
        eval_records,
        {record.case_id: cleaned_signal_frames[record.case_id] for record in eval_records},
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
    result = (
        pred_df.groupby(["case_id", "file_name", "wind_speed", "rpm"], as_index=False)["pred_wind_speed"]
        .mean()
        .rename(columns={"wind_speed": "true_wind_speed"})
    )
    return result


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
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
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


def merge_window_predictions(prediction_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    base = None
    for label, df in prediction_frames.items():
        block = df.rename(columns={"pred_wind_speed": f"pred_{label}"})
        if base is None:
            base = block
        else:
            base = base.merge(block[["case_id", f"pred_{label}"]], on="case_id", how="left")
    assert base is not None
    base["pred_2s_8s_fusion"] = (base["pred_2s"] + base["pred_8s"]) / 2.0
    for col in ["pred_2s", "pred_5s", "pred_8s", "pred_2s_8s_fusion"]:
        suffix = col.replace("pred_", "")
        base[f"signed_error_{suffix}"] = base[col] - base["true_wind_speed"]
        base[f"abs_error_{suffix}"] = base[f"signed_error_{suffix}"].abs()
    return base.sort_values("case_id").reset_index(drop=True)


def build_summary(added_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_name in ["2s", "5s", "8s", "2s_8s_fusion"]:
        rows.append(
            {
                "group_name": "added_cases",
                "model_name": model_name,
                "case_mae": float(added_df[f"abs_error_{model_name}"].mean()),
                "case_rmse": float(np.sqrt(np.mean(np.square(added_df[f"signed_error_{model_name}"])))),
                "case_count": int(len(added_df)),
            }
        )
    return pd.DataFrame(rows)


def write_summary_markdown(output_path: Path, added_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    lines = [
        "# added 外部验证（包含难工况训练池）",
        "",
        "- 训练池：`final` 全部带标签工况",
        "- 外部验证池：`added case 21-24`",
        "",
        "## added 外部验证",
        "",
    ]
    for _, row in added_df.iterrows():
        lines.append(
            f"- `工况{int(row['case_id'])}`: true=`{row['true_wind_speed']:.4f}`, pred_5s=`{row['pred_5s']:.4f}`, pred_fusion=`{row['pred_2s_8s_fusion']:.4f}`"
        )
    lines.extend(["", "## 汇总", ""])
    for _, row in summary_df.iterrows():
        lines.append(
            f"- `{row['model_name']}`: case_mae=`{row['case_mae']:.4f}`, case_rmse=`{row['case_rmse']:.4f}`"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
